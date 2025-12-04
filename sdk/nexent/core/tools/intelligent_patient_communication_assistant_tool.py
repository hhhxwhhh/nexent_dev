import json
import logging
import requests
from typing import List, Dict, Any, Optional
from pydantic import Field
from smolagents.tools import Tool

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("intelligent_patient_communication_assistant")


class IntelligentPatientCommunicationAssistantTool(Tool):
    """智能患者沟通助手工具，通过多模态交互帮助医生更好地与患者沟通病情"""

    name = "intelligent_patient_communication_assistant"
    description = (
        "基于医学诊断和患者信息，使用大模型API生成患者友好的沟通方案。"
        "该工具可以将复杂的医学概念转化为患者容易理解的语言，并提供个性化的沟通建议。"
        "支持结合医学图像进行更直观的解释。"
    )
    inputs = {
        "medical_condition": {
            "type": "string",
            "description": "医学诊断或病情描述"
        },
        "medical_images": {
            "type": "array",
            "description": "医学图像列表，每个图像包含data(图像base64数据)和type(图像类型)",
            "nullable": True
        },
        "patient_age": {
            "type": "integer",
            "description": "患者年龄",
            "nullable": True
        },
        "patient_gender": {
            "type": "string",
            "description": "患者性别",
            "nullable": True
        },
        "patient_concerns": {
            "type": "array",
            "description": "患者关心的问题列表",
            "items": {
                "type": "string"
            },
            "nullable": True
        },
        "communication_style": {
            "type": "string",
            "description": "沟通风格（如温和耐心、专业严谨、简洁明了等）",
            "default": "温和耐心",
            "nullable": True
        },
        "language": {
            "type": "string",
            "description": "使用的语言",
            "default": "中文",
            "nullable": True
        }
    }
    output_type = "object"
    category = ToolCategory.MULTIMODAL.value
    tool_sign = ToolSign.MULTIMODAL_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True),
        openkey_api_key: str = Field(description="OpenKey API密钥", default="sk-me3TxKSeoFk66yYDB5F24cF0244043558b29774bD4C1C357", exclude=True),
        openkey_base_url: str = Field(description="OpenKey API基础URL", default="https://openkey.cloud/v1", exclude=True),
        model_name: str = Field(description="使用的模型名称", default="gpt-4o-mini", exclude=True)
    ):
        super().__init__()
        self.observer = observer
        self.openkey_api_key = openkey_api_key
        self.openkey_base_url = openkey_base_url
        self.model_name = model_name
        self.running_prompt_zh = "正在生成智能患者沟通方案..."
        self.running_prompt_en = "Generating intelligent patient communication plan..."

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "comments", "text": "Intelligent Patient Communication Assistant"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _call_openkey_api(self, messages_content: List[Dict[str, Any]]) -> str:
        """调用OpenKey大模型API"""
        headers = {
            "Authorization": f"Bearer {self.openkey_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": f"你是一位专业的医患沟通专家，具备丰富的临床经验和出色的沟通技巧。你的任务是将复杂的医学概念用{self.inputs.get('language', {}).get('default', '中文')}转化为患者容易理解的语言，并根据患者的具体情况和担忧提供个性化的沟通建议。"
                },
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            "temperature": 0.5,
            "max_tokens": 1200
        }
        
        try:
            response = requests.post(
                f"{self.openkey_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
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
        medical_condition: str,
        medical_images: List[Dict[str, str]] = None,
        patient_age: int = None,
        patient_gender: str = None,
        patient_concerns: List[str] = None,
        communication_style: str = "温和耐心",
        language: str = "中文"
    ) -> Dict[str, Any]:
        """执行智能患者沟通助手"""
        try:
            self._send_tool_message()
            
            # 构建用户提示
            user_prompt = f"请根据以下医学信息，为医生提供一套有效的患者沟通方案：\n\n"
            user_prompt += f"医学诊断：{medical_condition}\n"
            
            if patient_age:
                user_prompt += f"患者年龄：{patient_age}岁\n"
                
            if patient_gender:
                user_prompt += f"患者性别：{patient_gender}\n"
                
            if patient_concerns:
                user_prompt += f"患者主要担忧：{', '.join(patient_concerns)}\n"
                
            user_prompt += f"期望沟通风格：{communication_style}\n"
            user_prompt += f"使用语言：{language}\n"
                
            user_prompt += "\n请按以下结构提供沟通方案：\n"
            user_prompt += "1. 病情通俗解释\n"
            user_prompt += "2. 关键医学概念图解建议（如需要）\n"
            user_prompt += "3. 针对患者担忧的回应要点\n"
            user_prompt += "4. 沟通要点和注意事项\n"
            user_prompt += "5. 患者教育建议\n"
            user_prompt += "6. 后续随访沟通建议\n\n"
            user_prompt += "请确保内容通俗易懂、富有同理心，并充分体现所选的沟通风格。"
            
            # 构造API请求消息
            messages_content = [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
            
            # 添加图像内容（如果有）
            if medical_images:
                for img in medical_images:
                    messages_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img.get('type', 'jpeg').lower()};base64,{img['data']}"
                        }
                    })
            
            # 调用大模型API
            communication_plan = self._call_openkey_api(messages_content)
            
            result = {
                "status": "success",
                "medical_condition": medical_condition,
                "communication_plan": communication_plan,
                "communication_style": communication_style,
                "additional_info": "此沟通方案仅供参考，实际沟通过程中请根据患者反应灵活调整。"
            }
            
            if patient_age:
                result["patient_age"] = patient_age
                
            if patient_gender:
                result["patient_gender"] = patient_gender
                
            if patient_concerns:
                result["patient_concerns"] = patient_concerns
                
            return result
            
        except Exception as e:
            logger.error(f"智能患者沟通助手工具执行失败: {str(e)}")
            return {
                "status": "error",
                "message": f"生成沟通方案失败: {str(e)}",
                "medical_condition": medical_condition
            }