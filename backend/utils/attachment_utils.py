from consts.const import LANGUAGE, MODEL_CONFIG_MAPPING
from typing import Union, BinaryIO

from jinja2 import Template, StrictUndefined
from nexent.core import MessageObserver
from nexent.core.models.openai_long_context_model import OpenAILongContextModel
from nexent.core.models.openai_vlm import OpenAIVLModel

from utils.config_utils import get_model_name_from_config, tenant_config_manager
from utils.prompt_template_utils import get_analyze_file_prompt_template


def convert_image_to_text(query: str, image_input: Union[str, BinaryIO], tenant_id: str, language: str = LANGUAGE["ZH"]):
    """
    Convert image to text description based on user query

    Args:
        query: User's question
        image_input: Image input (file path or binary data)
        tenant_id: Tenant ID for model configuration
        language: Language code ('zh' for Chinese, 'en' for English)

    Returns:
        str: Image description text
    """
    # 检查是否是病理学相关内容
    is_pathology_related = any(keyword in query.lower() for keyword in 
                              ['病理', '组织', '细胞', '肿瘤', '癌症', '病变', '切片'])
    
    vlm_model_config = tenant_config_manager.get_model_config(
        key=MODEL_CONFIG_MAPPING["vlm"], tenant_id=tenant_id)
    image_to_text_model = OpenAIVLModel(
        observer=MessageObserver(),
        model_id=get_model_name_from_config(
            vlm_model_config) if vlm_model_config else "",
        api_base=vlm_model_config.get("base_url", ""),
        api_key=vlm_model_config.get("api_key", ""),
        temperature=0.7,
        top_p=0.7,
        frequency_penalty=0.5,
        max_tokens=512
    )

    # 根据内容类型选择合适的模板
    if is_pathology_related:
        # 使用病理学专用模板
        from utils.prompt_template_utils import get_prompt_template
        prompts = get_prompt_template('pathology_document_analysis', language)
    else:
        # 使用通用模板
        prompts = get_analyze_file_prompt_template(language)
        
    system_prompt = Template(prompts['image_analysis']['system_prompt'],
                             undefined=StrictUndefined).render({'query': query})

    return image_to_text_model.analyze_image(image_input=image_input, system_prompt=system_prompt).content


def convert_long_text_to_text(query: str, file_context: str, tenant_id: str, language: str = LANGUAGE["ZH"]):
    """
    Convert long text to summarized text based on user query

    Args:
        query: User's question
        file_context: Text content to be analyzed
        tenant_id: Tenant ID for model configuration
        language: Language code ('zh' for Chinese, 'en' for English)

    Returns:
        tuple[str, Optional[float]]: Summarized text and truncation percentage (if any)
    """
    # 检查是否是病理学相关内容
    is_pathology_related = any(keyword in query.lower() for keyword in 
                              ['病理', '组织', '细胞', '肿瘤', '癌症', '病变', '切片'])
    
    main_model_config = tenant_config_manager.get_model_config(
        key=MODEL_CONFIG_MAPPING["llm"], tenant_id=tenant_id)
    long_text_to_text_model = OpenAILongContextModel(
        observer=MessageObserver(),
        model_id=get_model_name_from_config(main_model_config),
        api_base=main_model_config.get("base_url"),
        api_key=main_model_config.get("api_key"),
        max_context_tokens=main_model_config.get("max_tokens")
    )

    # 根据内容类型选择合适的模板
    if is_pathology_related:
        # 使用病理学专用模板
        from utils.prompt_template_utils import get_prompt_template
        prompts = get_prompt_template('pathology_document_analysis', language)
    else:
        # 使用通用模板
        prompts = get_analyze_file_prompt_template(language)
        
    system_prompt = Template(prompts['long_text_analysis']['system_prompt'],
                             undefined=StrictUndefined).render({'query': query})
    user_prompt = Template(
        prompts['long_text_analysis']['user_prompt'], undefined=StrictUndefined).render({})

    result, truncation_percentage = long_text_to_text_model.analyze_long_text(
        file_context, system_prompt, user_prompt)
    return result.content, truncation_percentage
