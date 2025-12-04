import json
import logging
import requests
from typing import List, Dict, Any
from pydantic import Field
from smolagents.tools import Tool

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("medical_terminology_standardizer")


class MedicalTerminologyStandardizerTool(Tool):
    """医学术语标准化工具，调用大模型API进行术语转换"""

    name = "medical_terminology_standardizer"
    description = (
        "将用户输入的非标准医学术语转换为标准医学术语。"
        "使用大模型API智能理解和转换术语，确保术语的一致性和准确性。"
        "特别适用于病理学、诊断学等领域。"
    )
    inputs = {
        "user_input": {
            "type": "string",
            "description": "用户的原始输入，可能包含非标准医学术语"
        },
        "target_domain": {
            "type": "string",
            "description": "目标医学领域 (如: 病理学, 诊断学, 解剖学等)",
            "default": "病理学",
            "nullable": True
        },
        "context": {
            "type": "string",
            "description": "上下文信息，有助于更准确的术语转换",
            "nullable": True
        }
    }
    output_type = "object"
    category = ToolCategory.UTILITIES.value
    tool_sign = ToolSign.SYSTEM_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True),
        openkey_api_key: str = Field(description="OpenKey API密钥", default="sk-me3TxKSeoFk66yXDB5F24cF0244043558b29774bD4C1C356", exclude=True),
        openkey_base_url: str = Field(description="OpenKey API基础URL", default="https://openkey.cloud/v1", exclude=True),
        model_name: str = Field(description="使用的模型名称", default="gpt-4o-mini", exclude=True)
    ):
        super().__init__()
        self.observer = observer
        self.openkey_api_key = openkey_api_key
        self.openkey_base_url = openkey_base_url
        self.model_name = model_name
        self.running_prompt_zh = "正在进行医学术语标准化..."
        self.running_prompt_en = "Standardizing medical terminology..."

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "book-medical", "text": "Medical Terminology Standardization"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _call_openkey_api(self, prompt: str) -> str:
        """调用OpenKey大模型API"""
        headers = {
            "Authorization": f"Bearer {self.openkey_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一位专业的医学术语专家，尤其擅长病理学领域。你的任务是将用户输入中的非标准医学术语转换为标准术语。请只输出转换后的标准术语，不要添加解释或其他内容。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                f"{self.openkey_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"OpenKey API调用失败: {response.status_code} - {response.text}")
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            logger.error(f"调用OpenKey API时发生错误: {str(e)}")
            return f"调用API时发生错误: {str(e)}"

    def forward(
        self,
        user_input: str,
        target_domain: str = "病理学",
        context: str = None
    ) -> Dict[str, Any]:
        """执行医学术语标准化"""
        try:
            self._send_tool_message()
            
            # 构建提示词
            prompt = f"请将以下输入中的非标准医学术语转换为{target_domain}领域的标准术语:\n\n"
            prompt += f"输入: {user_input}\n\n"
            
            if context:
                prompt += f"上下文: {context}\n\n"
            
            prompt += "请只输出转换后的标准术语，不需要解释。如果输入已经是标准术语或者无法识别，请原样返回。"
            
            # 调用大模型API
            standardized_term = self._call_openkey_api(prompt)
            
            result = {
                "status": "success",
                "original_input": user_input,
                "standardized_term": standardized_term,
                "target_domain": target_domain
            }
            
            return result
            
        except Exception as e:
            logger.error(f"医学术语标准化工具执行失败: {str(e)}")
            return {
                "status": "error",
                "message": f"术语标准化失败: {str(e)}",
                "original_input": user_input
            }