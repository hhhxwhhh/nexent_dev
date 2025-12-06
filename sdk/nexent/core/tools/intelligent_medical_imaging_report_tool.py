import json
import logging
import requests
from typing import List, Dict, Any
from pydantic import Field
from smolagents.tools import Tool

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("intelligent_medical_imaging_report")

class IntelligentMedicalImagingReportTool(Tool):
    """智能医疗影像报告生成工具，调用大模型API生成专业医疗影像报告"""

    name = "intelligent_medical_imaging_report"
    description = (
        "基于医疗影像和相关描述，使用大模型API生成专业的医疗影像报告。"
        "该工具可以分析影像内容、提供影像学表现描述、诊断意见和建议。"
        "适用于X光、CT、MRI等多种医疗影像类型。"
    )
    inputs = {
        "imaging_findings": {
            "type": "string",
            "description": "医疗影像的发现描述"
        },
        "imaging_type": {
            "type": "string",
            "description": "影像类型 (如: X光, CT, MRI, 超声等)",
            "nullable": True
        },
        "clinical_information": {
            "type": "string",
            "description": "临床信息 (症状、病史等)",
            "nullable": True
        },
        "patient_age": {
            "type": "integer",
            "description": "患者年龄",
            "nullable": True
        },
        "patient_gender": {
            "type": "string",
            "description": "患者性别 (male/female)",
            "nullable": True
        }
    }
    output_type = "object"
    category = ToolCategory.MULTIMODAL.value
    tool_sign = ToolSign.MULTIMODAL_OPERATION.value

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
        self.running_prompt_zh = "正在生成智能医疗影像报告..."
        self.running_prompt_en = "Generating intelligent medical imaging report..."

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "file-medical", "text": "Intelligent Medical Imaging Report"}]
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
                {"role": "system", "content": "你是一位经验丰富的放射科医生。根据提供的医疗影像发现和其他相关信息，生成一份专业的医疗影像报告。报告应包括影像学表现、诊断意见和建议。请以结构化、清晰的方式输出报告。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
            "enable_thinking": False
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
        imaging_findings: str,
        imaging_type: str = None,
        clinical_information: str = None,
        patient_age: int = None,
        patient_gender: str = None
    ) -> Dict[str, Any]:
        """执行智能医疗影像报告生成"""
        try:
            self._send_tool_message()
            
            # 构建提示词
            prompt = "请根据以下信息生成一份专业的医疗影像报告:\n\n"
            
            if imaging_type:
                prompt += f"影像类型: {imaging_type}\n"
                
            prompt += f"影像发现: {imaging_findings}\n\n"
            
            if patient_age:
                prompt += f"患者年龄: {patient_age}\n"
                
            if patient_gender:
                prompt += f"患者性别: {patient_gender}\n"
                
            if clinical_information:
                prompt += f"临床信息: {clinical_information}\n\n"
            
            prompt += (
                "请生成以下格式的报告:\n"
                "1. 影像学表现\n"
                "2. 诊断意见\n"
                "3. 建议\n\n"
                "报告应专业、准确、清晰，符合医疗规范。"
            )
            
            # 调用大模型API
            imaging_report = self._call_openkey_api(prompt)
            
            result = {
                "status": "success",
                "imaging_findings": imaging_findings,
                "imaging_report": imaging_report,
                "additional_info": "此报告仅供参考，不能替代专业医生的诊断。请结合临床情况进行综合判断。"
            }
            
            if imaging_type:
                result["imaging_type"] = imaging_type
                
            if clinical_information:
                result["clinical_information"] = clinical_information
                
            if patient_age:
                result["patient_age"] = patient_age
                
            if patient_gender:
                result["patient_gender"] = patient_gender
            
            return result
            
        except Exception as e:
            logger.error(f"智能医疗影像报告工具执行失败: {str(e)}")
            return {
                "status": "error",
                "message": f"生成医疗影像报告失败: {str(e)}",
                "imaging_findings": imaging_findings
            }