import json
import logging
import base64
import requests
from typing import List, Optional, Dict, Any
from io import BytesIO
from pydantic import Field
from smolagents.tools import Tool
from PIL import Image

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("pathology_image_analysis_tool")


class PathologyImageAnalysisTool(Tool):
    """病理图像分析工具（使用API进行分析）"""
    
    name = "pathology_image_analysis"
    description = (
        "专业的病理图像分析工具，可以分析组织切片、细胞图像等病理学相关图像。"
        "支持多种分析模式，包括肿瘤检测、细胞计数、组织分类等。"
        "适用于癌症筛查、病理诊断辅助等医疗场景。使用此工具时需要提供Base64编码的图像数据。"
    )
    inputs = {
        "image_data": {
            "type": "string",
            "description": "Base64编码的病理图像数据"
        },
        "analysis_type": {
            "type": "string",
            "description": "分析类型: tumor_detection(肿瘤检测), cell_count(细胞计数), tissue_classification(组织分类)",
            "default": "tumor_detection",
            "nullable": True
        },
        "confidence_threshold": {
            "type": "number",
            "description": "置信度阈值，低于此值的结果将被过滤",
            "default": 0.5,
            "nullable": True
        },
        "include_visualization": {
            "type": "boolean",
            "description": "是否包含可视化结果",
            "default": True,
            "nullable": True
        },
        "clinical_information": {
            "type": "string",
            "description": "临床信息（症状、病史等）",
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
        }
    }
    output_type = "string"
    category = ToolCategory.MULTIMODAL.value
    tool_sign = ToolSign.MULTIMODAL_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True),
        openkey_api_key: str = Field(description="OpenKey API密钥", default="sk-me3TxKSeoFk66yYDB5F24cF0244043558b29774bD4C1C357", exclude=True),
        openkey_base_url: str = Field(description="OpenKey API基础URL", default="https://openkey.cloud/v1", exclude=True),
        model_name: str = Field(description="使用的模型名称", default="gpt-4o-mini", exclude=True)
    ):
        """初始化病理图像分析工具（使用API进行分析）。
        
        Args:
            observer (MessageObserver, optional): 消息观察者实例。默认为 None。
            openkey_api_key (str): OpenKey API密钥。
            openkey_base_url (str): OpenKey API基础URL。
            model_name (str): 使用的模型名称。
        """
        super().__init__()
        self.observer = observer
        # 处理可能传入的Field对象，确保获取实际的默认值
        self.openkey_api_key = openkey_api_key if isinstance(openkey_api_key, str) else (getattr(openkey_api_key, 'default', None) or "sk-me3TxKSeoFk66yYDB5F24cF0244043558b29774bD4C1C357")
        self.openkey_base_url = openkey_base_url if isinstance(openkey_base_url, str) else (getattr(openkey_base_url, 'default', None) or "https://openkey.cloud/v1")
        self.model_name = model_name if isinstance(model_name, str) else (getattr(model_name, 'default', None) or "gpt-4o-mini")
        self.running_prompt_zh = "正在进行病理图像分析..."
        self.running_prompt_en = "Analyzing pathology image..."
        
        logger.info("PathologyImageAnalysisTool initialized with API-based analysis")

    def _send_tool_message(self, analysis_type: str):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "microscope", "text": f"分析病理图像 ({analysis_type})"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _call_openkey_api(self, image_data: str, analysis_type: str, clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        """调用OpenKey视觉语言模型API进行病理图像分析"""
        headers = {
            "Authorization": f"Bearer {self.openkey_api_key}",
            "Content-Type": "application/json"
        }
        
        # 构造针对不同类型图像的提示词
        analysis_prompts = {
            "tumor_detection": (
                "这是一张病理图像，请以病理学专家的角度详细分析是否存在肿瘤细胞或可疑区域。\n\n"
                "请按以下结构进行分析：\n"
                "1. 细胞形态学特征：细胞大小、形状、核质比、核仁特征等\n"
                "2. 组织结构：腺体结构、细胞排列方式、间质反应等\n"
                "3. 核特征：核大小、染色质分布、核分裂象等\n"
                "4. 病变范围和分布：病变区域占整个视野的比例、分布特点\n"
                "5. 可疑区域标记：指出任何可疑的区域及其特征\n"
                "6. 鉴别诊断：可能需要鉴别的其他病变类型\n"
                "7. 置信度评估：对分析结果的可信度进行评估\n"
                "8. 临床建议：建议的进一步检查或处理方式"
            ),
            "cell_count": (
                "这是一张细胞学图像，请精确分析并计数不同类型的细胞。\n\n"
                "请按以下结构进行分析：\n"
                "1. 细胞类型识别：识别并分类图像中的不同细胞类型\n"
                "2. 细胞计数：对各类细胞进行精确计数，提供绝对数量和百分比\n"
                "3. 细胞形态学分析：详细描述各类细胞的形态特征\n"
                "4. 异常细胞识别：识别任何形态异常的细胞及其特征\n"
                "5. 细胞活性评估：评估细胞的存活状态\n"
                "6. 统计数据：提供细胞分布的统计数据\n"
            ),
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["tumor_detection"])
        
        # 添加临床上下文信息
        if clinical_context:
            prompt += f"\n\n临床信息："
            if clinical_context.get("patient_age"):
                prompt += f"\n患者年龄：{clinical_context['patient_age']}岁"
            if clinical_context.get("patient_gender"):
                prompt += f"\n患者性别：{clinical_context['patient_gender']}"
            if clinical_context.get("clinical_information"):
                prompt += f"\n临床症状：{clinical_context['clinical_information']}"
            
        prompt += "\n\n请提供以下信息：\n"
        prompt += "1. 主要发现\n"
        prompt += "2. 置信度评估\n"
        prompt += "3. 可能的诊断\n"
        prompt += "4. 建议的进一步检查\n"
        prompt += "5. 注意事项\n\n"
        prompt += "请以结构化、清晰的方式回答，强调这只是辅助建议，不能替代专业医生的面诊。"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "enable_thinking": False
        }
        
        try:
            response = requests.post(
                f"{self.openkey_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "content": result["choices"][0]["message"]["content"]
                }
            else:
                logger.error(f"OpenKey API调用失败: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"API调用失败: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"调用OpenKey API时发生错误: {str(e)}")
            return {
                "status": "error",
                "message": f"调用API时发生错误: {str(e)}"
            }

    def _decode_image_to_pil(self, image_data: str) -> Image.Image:
        """解码Base64图像数据为PIL图像。
        
        Args:
            image_data (str): Base64编码的图像数据。
            
        Returns:
            PIL.Image: 解码后的图像。
        """
        try:
            # 解码Base64数据
            image_bytes = base64.b64decode(image_data)
            image_pil = Image.open(BytesIO(image_bytes))
            
            # 确保图像是RGB格式
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            logger.info(f"Image decoded successfully. Size: {image_pil.size}")
            return image_pil
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            raise Exception(f"图像数据解码失败: {str(e)}")

    def _create_visualization(self, image_pil: Image.Image, analysis_type: str) -> str:
        """创建可视化结果。
        
        Args:
            image_pil (Image.Image): 原始PIL图像。
            analysis_type (str): 分析类型。
            
        Returns:
            str: Base64编码的可视化图像。
        """
        # 将PIL图像转换为OpenCV格式
        import cv2
        import numpy as np
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 添加文本注释
        cv2.putText(image_cv, f"Analysis: {analysis_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 将可视化图像转换为Base64
        vis_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        vis_pil.save(buffer, format="PNG")
        visualization_data = base64.b64encode(buffer.getvalue()).decode()
        
        return visualization_data

    def forward(
        self, 
        image_data: str, 
        analysis_type: str = "tumor_detection",
        confidence_threshold: float = 0.5,
        include_visualization: bool = True,
        clinical_information: str = None,
        patient_age: int = None,
        patient_gender: str = None
    ) -> str:
        """分析病理图像的主要方法。
        
        Args:
            image_data (str): Base64编码的病理图像数据。
            analysis_type (str): 分析类型。
            confidence_threshold (float): 置信度阈值。
            include_visualization (bool): 是否包含可视化结果。
            clinical_information (str): 临床信息。
            patient_age (int): 患者年龄。
            patient_gender (str): 患者性别。
            
        Returns:
            str: JSON格式的分析结果。
            
        Raises:
            Exception: 如果图像数据无效或其他错误。
        """
        try:
            # 发送工具运行消息
            self._send_tool_message(analysis_type)

            # 解码图像数据以验证其有效性
            image_pil = self._decode_image_to_pil(image_data)
            
            # 构建临床上下文
            clinical_context = {
                "clinical_information": clinical_information,
                "patient_age": patient_age,
                "patient_gender": patient_gender
            }
            
            # 调用API进行分析
            api_result = self._call_openkey_api(image_data, analysis_type, clinical_context)
            
            if api_result["status"] == "error":
                raise Exception(api_result["message"])
            
            # 准备结果
            result = {
                "status": "success",
                "analysis_type": analysis_type,
                "confidence_threshold": confidence_threshold,
                "findings": api_result["content"],
                "additional_info": "此分析仅供参考，不能替代专业医生的诊断。如有严重症状，请立即就医。"
            }
            
            # 添加可视化结果（如果需要）
            if include_visualization:
                result["visualization"] = self._create_visualization(image_pil, analysis_type)
            
            # 添加临床信息到结果中
            if clinical_information:
                result["clinical_information"] = clinical_information
                
            if patient_age:
                result["patient_age"] = patient_age
                
            if patient_gender:
                result["patient_gender"] = patient_gender
            
            logger.info(f"Pathology image analysis completed for type: {analysis_type}")
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in pathology image analysis: {str(e)}", exc_info=True)
            error_msg = f"病理图像分析失败: {str(e)}"
            raise Exception(error_msg)