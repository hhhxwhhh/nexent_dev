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
    """获取当前时间工具，结合季节和时间提供病理学相关信息"""
    
    name = "get_current_time"
    description = (
        "获取当前的日期和时间信息，并结合季节和月份提供病理学相关的实用信息。"
        "包括季节性疾病模式、标本保存注意事项、实验室工作环境要求等。"
        "适用于需要获取当前时间戳、日期或其他时间相关信息的场景，并提供额外的病理学价值信息。"
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
        },
        "include_pathology_info": {
            "type": "boolean",
            "description": "是否包含与病理学相关的时间和季节信息，如季节性疾病模式、标本保存建议等。",
            "default": True,
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
        self.running_prompt_zh = "正在获取当前时间及相关病理学信息..."
        self.running_prompt_en = "Getting current time and related pathology information..."

    def forward(self, timezone: str = "UTC", format: str = "%Y-%m-%d %H:%M:%S %Z", include_pathology_info: bool = True) -> str:
        """获取当前时间的主要方法。
        
        Args:
            timezone (str, optional): 时区标识符。默认为 "UTC"。
            format (str, optional): 时间格式字符串。默认为 "%Y-%m-%d %H:%M:%S %Z"。
            include_pathology_info (bool, optional): 是否包含病理学相关信息。默认为 True。
            
        Returns:
            str: JSON格式的当前时间信息，可能包含病理学相关信息
            
        Raises:
            Exception: 如果时区无效或其他错误
        """
        try:
            # 发送工具运行消息
            if self.observer:
                running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
                self.observer.add_message("", ProcessType.TOOL, running_prompt)
                card_content = [{"icon": "clock", "text": f"获取 {timezone} 时间及相关信息"}]
                self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

            # 获取当前时间
            current_time_info = self._get_current_time(timezone, format)
            
            # 如果需要，添加病理学相关信息
            if include_pathology_info:
                pathology_info = self._get_pathology_related_info(current_time_info)
                current_time_info.update(pathology_info)
            
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
            
            # 添加病理学相关信息
            if include_pathology_info:
                success_msg.update({
                    "season": current_time_info.get("season"),
                    "seasonal_disease_patterns": current_time_info.get("seasonal_disease_patterns"),
                    "specimen_storage_considerations": current_time_info.get("specimen_storage_considerations"),
                    "lab_environment_requirements": current_time_info.get("lab_environment_requirements")
                })

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
                try:
                    import pytz
                    tz = pytz.timezone(timezone)
                    now = datetime.now(tz)
                    formatted_time = now.strftime(format)
                    iso_format = now.isoformat()
                    timestamp = now.timestamp()
                except ImportError:
                    # 如果pytz不可用，回退到简单处理
                    now = datetime.utcnow()
                    formatted_time = now.strftime(format.replace("%Z", "UTC"))
                    iso_format = now.isoformat() + "Z"
                    timestamp = now.timestamp()
                
        except Exception as e:
            # 如果时区无效，回退到UTC
            logger.warning(f"Timezone {timezone} not available, falling back to UTC: {str(e)}")
            now = datetime.utcnow()
            formatted_time = now.strftime(format.replace("%Z", "UTC"))
            iso_format = now.isoformat() + "Z"
            timestamp = now.timestamp()
            
        return {
            "formatted_time": formatted_time,
            "timestamp": timestamp,
            "iso_format": iso_format,
            "month": now.month,
            "day_of_year": now.timetuple().tm_yday
        }
        
    def _get_pathology_related_info(self, time_info: dict) -> dict:
        """根据当前时间提供病理学相关信息。
        
        Args:
            time_info (dict): 包含时间信息的字典
            
        Returns:
            dict: 包含病理学相关信息的字典
        """
        month = time_info["month"]
        day_of_year = time_info["day_of_year"]
        
        # 确定季节
        if month in [12, 1, 2]:
            season = "winter"
            season_name = "冬季"
        elif month in [3, 4, 5]:
            season = "spring"
            season_name = "春季"
        elif month in [6, 7, 8]:
            season = "summer"
            season_name = "夏季"
        else:  # 9, 10, 11
            season = "autumn"
            season_name = "秋季"
            
        # 季节性疾病模式
        seasonal_disease_patterns = self._get_seasonal_disease_patterns(season)
        
        # 标本保存注意事项
        specimen_storage_considerations = self._get_specimen_storage_considerations(season)
        
        # 实验室环境要求
        lab_environment_requirements = self._get_lab_environment_requirements(season)
        
        return {
            "season": season_name,
            "seasonal_disease_patterns": seasonal_disease_patterns,
            "specimen_storage_considerations": specimen_storage_considerations,
            "lab_environment_requirements": lab_environment_requirements
        }
        
    def _get_seasonal_disease_patterns(self, season: str) -> dict:
        """获取季节性疾病模式信息。
        
        Args:
            season (str): 季节标识符
            
        Returns:
            dict: 季节性疾病模式信息
        """
        patterns = {
            "winter": {
                "description": "冬季常见疾病模式",
                "diseases": [
                    "呼吸道感染（流感、肺炎等）导致的继发性细菌性肺炎",
                    "心血管疾病发病率增加",
                    "季节性情感障碍相关的自杀风险",
                    "冻伤和低体温症相关病例"
                ],
                "pathology_notes": "注意呼吸道标本的及时处理，避免病毒失活；关注心脏猝死病例的组织学特征"
            },
            "spring": {
                "description": "春季常见疾病模式",
                "diseases": [
                    "过敏性疾病（花粉症）导致的鼻炎和哮喘加重",
                    "传染病复苏（如结核病）",
                    "精神疾病复发增加"
                ],
                "pathology_notes": "关注过敏性炎症的组织学特征；注意传染病的免疫组化检测"
            },
            "summer": {
                "description": "夏季常见疾病模式",
                "diseases": [
                    "热射病和中暑相关死亡",
                    "皮肤癌筛查需求增加",
                    "食物中毒和胃肠道感染",
                    "虫媒传播疾病（如疟疾、登革热）"
                ],
                "pathology_notes": "注意高温对组织样本的影响；加强皮肤病变的组织学分析"
            },
            "autumn": {
                "description": "秋季常见疾病模式",
                "diseases": [
                    "呼吸系统疾病复发",
                    "抑郁症和其他情绪障碍",
                    "意外伤害增加（开学季交通事故等）"
                ],
                "pathology_notes": "关注慢性阻塞性肺疾病的组织学变化；注意创伤性损伤的法医病理学分析"
            }
        }
        
        return patterns.get(season, {})
        
    def _get_specimen_storage_considerations(self, season: str) -> dict:
        """获取标本保存注意事项。
        
        Args:
            season (str): 季节标识符
            
        Returns:
            dict: 标本保存注意事项
        """
        considerations = {
            "winter": {
                "temperature_control": "低温环境下注意防止样本冻结，尤其是液体样本",
                "transportation": "寒冷天气下运输时间可能延长，需加强保温措施",
                "special_precautions": "冰雪天气可能影响冷链物流，需提前规划"
            },
            "spring": {
                "temperature_control": "气温波动大，需要稳定的冷藏环境",
                "transportation": "雨季要注意防潮防水措施",
                "special_precautions": "湿度较大时需特别注意防止霉菌污染"
            },
            "summer": {
                "temperature_control": "高温天气下需要加强冷冻保存，防止蛋白质降解",
                "transportation": "炎热天气下需要更快的冷链运输",
                "special_precautions": "极端高温可能导致样本质量下降，需缩短处理时间"
            },
            "autumn": {
                "temperature_control": "温度适中，但仍需维持标准冷藏条件",
                "transportation": "干燥天气有利于样本运输",
                "special_precautions": "注意空调系统定期清洁，防止交叉污染"
            }
        }
        
        return considerations.get(season, {})
        
    def _get_lab_environment_requirements(self, season: str) -> dict:
        """获取实验室环境要求。
        
        Args:
            season (str): 季节标识符
            
        Returns:
            dict: 实验室环境要求
        """
        requirements = {
            "winter": {
                "humidity": "北方地区室内加热导致空气干燥，需保持适宜湿度（40-60%）",
                "equipment": "注意设备在低温下的正常运行，定期检查温控系统",
                "safety": "注意用电安全，防止因加热设备引起的火灾隐患"
            },
            "spring": {
                "humidity": "南方地区梅雨季节湿度高，需加强除湿措施",
                "equipment": "潮湿环境下注意仪器防潮，定期检查电路安全",
                "safety": "注意地面湿滑导致的安全隐患"
            },
            "summer": {
                "humidity": "高温高湿环境下需加强通风和除湿",
                "equipment": "注意散热，防止精密仪器过热损坏",
                "safety": "注意防暑降温，合理安排工作人员作息"
            },
            "autumn": {
                "humidity": "气候宜人，相对容易控制实验环境",
                "equipment": "适宜进行设备维护和校准",
                "safety": "注意落叶等杂物清理，保持通道畅通"
            }
        }
        
        return requirements.get(season, {})