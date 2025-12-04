import json
import logging
from io import BytesIO
from typing import List, Dict, Any, Optional
from jinja2 import Template, StrictUndefined
from pydantic import Field
from smolagents.tools import Tool

from nexent.core.models import OpenAIVLModel
from nexent.core.utils.observer import MessageObserver, ProcessType
from nexent.core.utils.prompt_template_utils import get_prompt_template
from nexent.core.utils.tools_common_message import ToolCategory, ToolSign
from nexent.storage import MinIOStorageClient
from nexent.multi_modal.load_save_object import LoadSaveObjectManager

logger = logging.getLogger("multimodal_qa_fusion_tool")


class MultimodalQAFusionTool(Tool):
    """多模态问答融合工具，结合图像和文本输入提供综合回答"""

    name = "multimodal_qa_fusion"
    description = (
        "结合图像和文本问题进行综合分析，提供更全面的答案。\n"
        "此工具专门用于处理病理学相关的多模态问答场景，能够同时分析图像内容和文本问题。\n"
        "支持多种图像来源，包括S3、HTTP和HTTPS URL。"
    )
    inputs = {
        "image_urls_list": {
            "type": "array",
            "description": "图像URL列表 (支持 S3、HTTP、HTTPS 协议)。例如: ['https://example.com/image.png']",
        },
        "query": {
            "type": "string",
            "description": "用户的问题文本"
        },
        "context": {
            "type": "string",
            "description": "额外的上下文信息，有助于更准确地回答问题",
            "nullable": True
        }
    }
    output_type = "object"
    category = ToolCategory.MULTIMODAL.value
    tool_sign = ToolSign.MULTIMODAL_OPERATION.value

    def __init__(
            self,
            observer: MessageObserver = Field(
                description="消息观察者",
                default=None,
                exclude=True),
            vlm_model: OpenAIVLModel = Field(
                description="视觉语言模型",
                default=None,
                exclude=True),
            storage_client: MinIOStorageClient = Field(
                description="存储客户端，用于下载S3、HTTP、HTTPS URL的文件",
                default=None,
                exclude=True)
    ):
        super().__init__()
        self.observer = observer
        self.vlm_model = vlm_model
        self.storage_client = storage_client
        # 创建LoadSaveObjectManager实例
        self.mm = LoadSaveObjectManager(storage_client=self.storage_client)

        # 动态应用load_object装饰器到forward方法
        self.forward = self.mm.load_object(input_names=["image_urls_list"])(self._forward_impl)

        self.running_prompt_zh = "正在进行多模态问答融合分析..."
        self.running_prompt_en = "Performing multimodal QA fusion analysis..."

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "microscope", "text": "Multimodal QA Fusion Analysis"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _forward_impl(self, image_urls_list: List[bytes], query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        结合图像和文本问题进行综合分析
        
        注意: 此方法由load_object装饰器包装，该装饰器会从URL下载图像并将其转换为字节传递给此方法。

        参数:
            image_urls_list: 由装饰器转换的图像字节列表
            query: 用户的问题文本
            context: 额外的上下文信息

        返回:
            Dict[str, Any]: 包含分析结果的字典

        异常:
            Exception: 如果图像无法下载或分析失败
        """
        try:
            # 发送工具运行消息
            self._send_tool_message()

            if image_urls_list is None:
                raise ValueError("image_urls_list cannot be None")

            if not isinstance(image_urls_list, list):
                raise ValueError("image_urls_list must be a list of bytes")

            if not image_urls_list:
                raise ValueError("image_urls_list must contain at least one image")

            # 准备系统提示词
            language = self.observer.lang if self.observer else "en"
            
            # 构造完整的提示词
            full_prompt = f"请仔细分析提供的医学图像并回答以下问题:\n\n问题: {query}"
            if context:
                full_prompt += f"\n\n上下文信息: {context}"

            analysis_results = []
            
            # 分析每张图像
            for index, image_bytes in enumerate(image_urls_list, start=1):
                logger.info(f"Analyzing image #{index} with query: {query}")
                image_stream = BytesIO(image_bytes)
                
                try:
                    # 使用视觉语言模型分析图像
                    response = self.vlm_model.analyze_image(
                        image_input=image_stream,
                        system_prompt=f"你是一个专业的病理学专家。{full_prompt}"
                    )
                    analysis_results.append({
                        "image_index": index,
                        "analysis": response.content
                    })
                except Exception as e:
                    error_msg = f"Error analyzing image {index}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    analysis_results.append({
                        "image_index": index,
                        "error": error_msg
                    })

            # 整合所有图像分析结果和问题，生成最终答案
            final_answer_prompt = f"""
你是一个专业的病理学专家。基于以下图像分析结果和用户问题，提供一个综合性的回答。

用户问题: {query}
{f"上下文信息: {context}" if context else ""}

图像分析结果:
"""
            
            for result in analysis_results:
                if "error" in result:
                    final_answer_prompt += f"[图像 {result['image_index']} 分析失败: {result['error']}]\n"
                else:
                    final_answer_prompt += f"[图像 {result['image_index']}]: {result['analysis']}\n"

            final_answer_prompt += "\n请结合以上所有信息，给出详细且专业的回答。"

            # 调用VLM模型生成最终答案
            final_messages = [
                {"role": "system", "content": "你是一个专业的病理学专家，能够结合图像分析结果和专业知识回答复杂问题。"},
                {"role": "user", "content": final_answer_prompt}
            ]
            
            final_response = self.vlm_model(messages=final_messages)

            return {
                "status": "success",
                "query": query,
                "context": context,
                "image_count": len(image_urls_list),
                "individual_analyses": analysis_results,
                "final_answer": final_response.content
            }

        except Exception as e:
            logger.error(f"Multimodal QA fusion tool execution failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"多模态问答融合分析失败: {str(e)}",
                "query": query,
                "context": context
            }