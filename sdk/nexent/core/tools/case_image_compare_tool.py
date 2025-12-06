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

logger = logging.getLogger("case_image_comparison_tool")


class CaseImageComparisonTool(Tool):
    """病例图像比较工具，用于比较多个医学图像并分析差异"""
    
    name = "case_image_comparison"
    description = (
        "专业的病例图像比较工具，可以比较两个或多个医学图像并分析它们之间的差异和相似性。"
        "支持多种医学图像类型，如X光、CT、MRI等。"
        "适用于病情跟踪、治疗效果评估等医疗场景。使用此工具时需要提供Base64编码的图像数据。"
    )
    inputs = {
        "image_data_list": {
            "type": "array",
            "description": "Base64编码的医学图像数据列表"
        },
        "analysis_type": {
            "type": "string",
            "description": "分析类型: bone_fracture(骨折), lung_nodule(肺结节), brain_mri(脑部MRI), general(通用)",
            "default": "general",
            "nullable": True
        },
        "patient_info": {
            "type": "object",
            "description": "患者信息，包括年龄、性别和症状",
            "properties": {
                "age": {"type": "integer", "description": "患者年龄"},
                "gender": {"type": "string", "description": "患者性别"},
                "symptoms": {"type": "string", "description": "患者症状"}
            },
            "nullable": True
        }
    }
    output_type = "string"
    category = ToolCategory.MULTIMODAL.value
    tool_sign = ToolSign.MULTIMODAL_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True),
        openkey_api_key: str = Field(description="OpenKey API密钥", default="sk-me3TxKSeoFk66yZDB5F24cF0244043558b29774bD4C1C358", exclude=True),
        openkey_base_url: str = Field(description="OpenKey API基础URL", default="https://openkey.cloud/v1", exclude=True),
        model_name: str = Field(description="使用的模型名称", default="gpt-4o-mini", exclude=True)
    ):
        """初始化病例图像比较工具。
        
        Args:
            observer (MessageObserver, optional): 消息观察者实例。默认为 None。
            openkey_api_key (str): OpenKey API密钥。
            openkey_base_url (str): OpenKey API基础URL。
            model_name (str): 使用的模型名称。
        """
        super().__init__()
        self.observer = observer
        self.openkey_api_key = openkey_api_key
        self.openkey_base_url = openkey_base_url
        self.model_name = model_name
        self.running_prompt_zh = "正在进行病例图像比较分析..."
        self.running_prompt_en = "Analyzing case images comparison..."
        
        logger.info("CaseImageComparisonTool initialized with API-based analysis")

    def _send_tool_message(self, analysis_type: str):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "images", "text": f"病例图像比较分析 ({analysis_type})"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _call_openkey_api(self, image_data_list: List[str], analysis_type: str, patient_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """调用OpenKey视觉语言模型API进行病例图像比较分析"""
        headers = {
            "Authorization": f"Bearer {self.openkey_api_key}",
            "Content-Type": "application/json"
        }
        
        # 构造针对不同类型图像的提示词
        analysis_prompts = {
            "bone_fracture": "这是几张骨科相关的医学图像，请分析骨折的情况及愈合过程。",
            "lung_nodule": "这是几张肺部结节的医学图像，请分析结节的变化情况。",
            "brain_mri": "这是几张脑部MRI图像，请分析病变区域的变化。",
            "general": "这是几张医学图像，请分析它们之间的异同点。"
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        # 添加患者信息到提示词
        if patient_info:
            prompt += f"\n\n患者信息："
            if patient_info.get("age"):
                prompt += f"\n年龄：{patient_info['age']}岁"
            if patient_info.get("gender"):
                prompt += f"\n性别：{patient_info['gender']}"
            if patient_info.get("symptoms"):
                prompt += f"\n症状：{patient_info['symptoms']}"
            
        prompt += f"\n\n请比较这{len(image_data_list)}张图像，提供以下信息：\n"
        prompt += "1. 图像间的相似性\n"
        prompt += "2. 图像间的主要差异\n"
        prompt += "3. 病情发展趋势分析\n"
        prompt += "4. 诊断建议\n"
        prompt += "5. 后续治疗或检查建议\n\n"
        prompt += "请以结构化、清晰的方式回答，强调这只是辅助建议，不能替代专业医生的面诊。"
        
        # 构造内容数组，包含所有图像
        content_array = [{"type": "text", "text": prompt}]
        for image_data in image_data_list:
            content_array.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_array
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
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

    def _decode_images_to_pil(self, image_data_list: List[str]) -> List[Image.Image]:
        """解码Base64图像数据为PIL图像列表。
        
        Args:
            image_data_list (List[str]): Base64编码的图像数据列表。
            
        Returns:
            List[PIL.Image]: 解码后的图像列表。
        """
        decoded_images = []
        for i, image_data in enumerate(image_data_list):
            try:
                # 解码Base64数据
                image_bytes = base64.b64decode(image_data)
                image_pil = Image.open(BytesIO(image_bytes))
                
                # 确保图像是RGB格式
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                
                logger.info(f"Image {i+1} decoded successfully. Size: {image_pil.size}")
                decoded_images.append(image_pil)
            except Exception as e:
                logger.error(f"Failed to decode image {i+1}: {str(e)}")
                raise Exception(f"图像数据 {i+1} 解码失败: {str(e)}")
        
        return decoded_images

    def forward(
        self, 
        image_data_list: List[str], 
        analysis_type: str = "general",
        patient_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """比较病例图像的主要方法。
        
        Args:
            image_data_list (List[str]): Base64编码的医学图像数据列表。
            analysis_type (str): 分析类型。
            patient_info (Optional[Dict[str, Any]]): 患者信息。
            
        Returns:
            str: JSON格式的分析结果。
            
        Raises:
            Exception: 如果图像数据无效或其他错误。
        """
        try:
            # 验证输入
            if not image_data_list or len(image_data_list) < 2:
                raise ValueError("至少需要提供2个图像进行比较")
            
            # 发送工具运行消息
            self._send_tool_message(analysis_type)

            # 解码图像数据以验证其有效性
            image_pil_list = self._decode_images_to_pil(image_data_list)
            
            # 调用API进行分析
            api_result = self._call_openkey_api(image_data_list, analysis_type, patient_info)
            
            if api_result["status"] == "error":
                raise Exception(api_result["message"])
            
            # 准备结果
            result = {
                "status": "success",
                "image_count": len(image_data_list),
                "analysis_type": analysis_type,
                "findings": api_result["content"],
                "additional_info": "此分析仅供参考，不能替代专业医生的诊断。如有严重症状，请立即就医。"
            }
            
            # 添加患者信息到结果中
            if patient_info:
                result["patient_info"] = patient_info
            
            logger.info(f"Case image comparison analysis completed for type: {analysis_type}")
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in case image comparison analysis: {str(e)}", exc_info=True)
            error_msg = f"病例图像比较分析失败: {str(e)}"
            raise Exception(error_msg)


class TreatmentProgressAnalysisTool(Tool):
    """治疗进度分析工具，用于分析治疗前后的图像变化"""
    
    name = "treatment_progress_analysis"
    description = (
        "专业的治疗进度分析工具，通过对比治疗前后的医学图像，分析治疗效果和病情进展。"
        "适用于各种需要跟踪治疗效果的医疗场景。使用此工具时需要提供Base64编码的图像数据。"
    )
    inputs = {
        "before_image_data": {
            "type": "string",
            "description": "治疗前Base64编码的图像数据"
        },
        "after_image_data": {
            "type": "string",
            "description": "治疗后Base64编码的图像数据"
        },
        "treatment_type": {
            "type": "string",
            "description": "治疗类型"
        },
        "time_interval": {
            "type": "string",
            "description": "治疗间隔时间",
            "nullable": True
        },
        "patient_info": {
            "type": "object",
            "description": "患者信息，包括年龄、性别和症状",
            "properties": {
                "age": {"type": "integer", "description": "患者年龄"},
                "gender": {"type": "string", "description": "患者性别"},
                "symptoms": {"type": "string", "description": "患者症状"}
            },
            "nullable": True
        }
    }
    output_type = "string"
    category = ToolCategory.MULTIMODAL.value
    tool_sign = ToolSign.MULTIMODAL_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True),
        openkey_api_key: str = Field(description="OpenKey API密钥", default="sk-me3TxKSeoFk66yZDB5F24cF0244043558b29774bD4C1C358", exclude=True),
        openkey_base_url: str = Field(description="OpenKey API基础URL", default="https://openkey.cloud/v1", exclude=True),
        model_name: str = Field(description="使用的模型名称", default="gpt-4o-mini", exclude=True)
    ):
        """初始化治疗进度分析工具。
        
        Args:
            observer (MessageObserver, optional): 消息观察者实例。默认为 None。
            openkey_api_key (str): OpenKey API密钥。
            openkey_base_url (str): OpenKey API基础URL。
            model_name (str): 使用的模型名称。
        """
        super().__init__()
        self.observer = observer
        self.openkey_api_key = openkey_api_key
        self.openkey_base_url = openkey_base_url
        self.model_name = model_name
        self.running_prompt_zh = "正在进行治疗进度分析..."
        self.running_prompt_en = "Analyzing treatment progress..."
        
        logger.info("TreatmentProgressAnalysisTool initialized with API-based analysis")

    def _send_tool_message(self, treatment_type: str):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "activity", "text": f"治疗进度分析 ({treatment_type})"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _call_openkey_api(self, before_image_data: str, after_image_data: str, treatment_type: str, time_interval: Optional[str], patient_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """调用OpenKey视觉语言模型API进行治疗进度分析"""
        headers = {
            "Authorization": f"Bearer {self.openkey_api_key}",
            "Content-Type": "application/json"
        }
        
        # 构造提示词
        prompt = f"这是同一患者治疗前后的两张医学图像，请分析治疗效果。\n\n"
        prompt += f"治疗类型: {treatment_type}\n"
        if time_interval:
            prompt += f"治疗间隔时间: {time_interval}\n"
            
        if patient_info:
            prompt += f"\n患者信息："
            if patient_info.get("age"):
                prompt += f"\n年龄：{patient_info['age']}岁"
            if patient_info.get("gender"):
                prompt += f"\n性别：{patient_info['gender']}"
            if patient_info.get("symptoms"):
                prompt += f"\n症状：{patient_info['symptoms']}"
        
        prompt += "\n\n请提供以下信息：\n"
        prompt += "1. 病变区域的变化情况\n"
        prompt += "2. 治疗效果评估\n"
        prompt += "3. 病情发展趋势\n"
        prompt += "4. 后续治疗建议\n\n"
        prompt += "请以结构化、清晰的方式回答，强调这只是辅助建议，不能替代专业医生的面诊。"
        
        # 构造内容数组
        content_array = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{before_image_data}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{after_image_data}"
                }
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_array
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1500
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

    def forward(
        self, 
        before_image_data: str,
        after_image_data: str,
        treatment_type: str,
        time_interval: Optional[str] = None,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """分析治疗进度的主要方法。
        
        Args:
            before_image_data (str): 治疗前Base64编码的图像数据。
            after_image_data (str): 治疗后Base64编码的图像数据。
            treatment_type (str): 治疗类型。
            time_interval (Optional[str]): 治疗间隔时间。
            patient_info (Optional[Dict[str, Any]]): 患者信息。
            
        Returns:
            str: JSON格式的分析结果。
            
        Raises:
            Exception: 如果图像数据无效或其他错误。
        """
        try:
            # 发送工具运行消息
            self._send_tool_message(treatment_type)

            # 解码图像数据以验证其有效性
            before_image_pil = self._decode_image_to_pil(before_image_data)
            after_image_pil = self._decode_image_to_pil(after_image_data)
            
            # 调用API进行分析
            api_result = self._call_openkey_api(before_image_data, after_image_data, treatment_type, time_interval, patient_info)
            
            if api_result["status"] == "error":
                raise Exception(api_result["message"])
            
            # 准备结果
            result = {
                "status": "success",
                "treatment_type": treatment_type,
                "time_interval": time_interval,
                "findings": api_result["content"],
                "additional_info": "此分析仅供参考，不能替代专业医生的诊断。请结合临床情况进行综合判断。"
            }
            
            # 添加患者信息到结果中
            if patient_info:
                result["patient_info"] = patient_info
            
            logger.info(f"Treatment progress analysis completed for type: {treatment_type}")
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in treatment progress analysis: {str(e)}", exc_info=True)
            error_msg = f"治疗进度分析失败: {str(e)}"
            raise Exception(error_msg)