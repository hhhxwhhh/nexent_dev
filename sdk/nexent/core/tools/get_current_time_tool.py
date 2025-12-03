import json 
import logging 
from datetime import datetime
from typing import Optional
from pydantic import Field
from smolagents.tools import Tool

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("get_current_time_tool")

class GetCurrentTimeTool(Tool):
    """获取当前时间工具"""
    
    name = "get_current_time"
    description = (
        "获取当前的日期和时间信息。"
        "可以指定时区获取对应的时间，如果不指定时区则默认返回UTC时间。"
        "适用于需要获取当前时间戳、日期或其他时间相关信息的场景。"
    )
    inputs = {
        "timezone": {
            "type": "string", 
            "description": "时区标识符，例如 'UTC', 'Asia/Shanghai', 'America/New_York' 等。如果未提供，默认为 UTC。",
            "default": "UTC",
            "nullable": True
        },
        "format": {
            "type": "string", 
            "description": "时间格式字符串，例如 '%Y-%m-%d %H:%M:%S'。如果未提供，则返回标准格式。",
            "default": "%Y-%m-%d %H:%M:%S %Z",
            "nullable": True
        }
    }
    output_type = "string"
    category = ToolCategory.UTILITIES.value
    tool_sign = ToolSign.SYSTEM_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True)
    ):
        """初始化获取当前时间工具。
        
        Args:
            observer (MessageObserver, optional): 消息观察者实例。默认为 None。
        """
        super().__init__()
        self.observer = observer
        self.running_prompt_zh = "正在获取当前时间..."
        self.running_prompt_en = "Getting current time..."

    def forward(self, timezone: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """获取当前时间的主要方法。
        
        Args:
            timezone (str, optional): 时区标识符。默认为 "UTC"。
            format (str, optional): 时间格式字符串。默认为 "%Y-%m-%d %H:%M:%S %Z"。
            
        Returns:
            str: JSON格式的当前时间信息
            
        Raises:
            Exception: 如果时区无效或其他错误
        """
        try:
            # 发送工具运行消息
            if self.observer:
                running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
                self.observer.add_message("", ProcessType.TOOL, running_prompt)
                card_content = [{"icon": "clock", "text": f"获取 {timezone} 时间"}]
                self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

            # 获取当前时间
            current_time_info = self._get_current_time(timezone, format)
            
            logger.info(f"Successfully got current time for timezone {timezone}")
            
            # 准备成功消息
            success_msg = {
                "status": "success",
                "timezone": timezone,
                "formatted_time": current_time_info["formatted_time"],
                "timestamp": current_time_info["timestamp"],
                "iso_format": current_time_info["iso_format"],
                "message": f"Successfully retrieved current time for {timezone}"
            }

            return json.dumps(success_msg, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error getting current time: {str(e)}", exc_info=True)
            error_msg = f"获取当前时间失败: {str(e)}"
            raise Exception(error_msg)

    def _get_current_time(self, timezone: str, format: str) -> dict:
        """获取指定时区的当前时间。
        
        Args:
            timezone (str): 时区标识符
            format (str): 时间格式字符串
            
        Returns:
            dict: 包含各种时间格式的字典
            
        Raises:
            Exception: 如果时区无效
        """
        try:
            # 对于UTC时区的特殊处理
            if timezone.upper() == "UTC":
                now = datetime.utcnow()
                formatted_time = now.strftime(format.replace("%Z", "UTC"))
                iso_format = now.isoformat() + "Z"
                timestamp = now.timestamp()
            else:
                # 对于其他时区，需要使用pytz或datetime.timezone
                # 这里简化处理，实际项目中可能需要引入pytz库
                import pytz
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
                formatted_time = now.strftime(format)
                iso_format = now.isoformat()
                timestamp = now.timestamp()
                
            return {
                "formatted_time": formatted_time,
                "timestamp": timestamp,
                "iso_format": iso_format
            }
        except Exception as e:
            # 如果pytz不可用或时区无效，回退到UTC
            logger.warning(f"Timezone {timezone} not available, falling back to UTC: {str(e)}")
            now = datetime.utcnow()
            formatted_time = now.strftime(format.replace("%Z", "UTC"))
            iso_format = now.isoformat() + "Z"
            timestamp = now.timestamp()
            
            return {
                "formatted_time": formatted_time,
                "timestamp": timestamp,
                "iso_format": iso_format
            }