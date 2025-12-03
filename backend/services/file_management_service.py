import asyncio
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, AsyncGenerator
import json
import uuid

from fastapi import UploadFile

from consts.const import UPLOAD_FOLDER, MAX_CONCURRENT_UPLOADS, MODEL_CONFIG_MAPPING, LANGUAGE
from database.attachment_db import (
    upload_fileobj,
    get_file_url,
    get_content_type,
    get_file_stream,
    delete_file,
    list_files
)
from services.vectordatabase_service import ElasticSearchService, get_vector_db_core
from utils.config_utils import tenant_config_manager, get_model_name_from_config
from utils.file_management_utils import save_upload_file, get_file_processing_messages_template
from utils.prompt_template_utils import get_file_processing_messages_template

from nexent import MessageObserver
from nexent.core.models import OpenAILongContextModel

# Create upload directory
upload_dir = Path(UPLOAD_FOLDER)
upload_dir.mkdir(exist_ok=True)
upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

logger = logging.getLogger("file_management_service")

# Simple preprocess task manager
class PreprocessManager:
    def __init__(self):
        self.tasks = {}
    
    def register_preprocess_task(self, task_id: str, conversation_id: int, task):
        self.tasks[task_id] = {
            "conversation_id": conversation_id,
            "task": task
        }

# Global instance
preprocess_manager = PreprocessManager()


async def upload_files_impl(destination: str, file: List[UploadFile], folder: str = None, index_name: Optional[str] = None) -> tuple:
    """
    Upload files to local storage or MinIO based on destination.

    Args:
        destination: "local" or "minio"
        file: List of UploadFile objects
        folder: Folder name for MinIO uploads

    Returns:
        tuple: (errors, uploaded_file_paths, uploaded_filenames)
    """
    uploaded_filenames = []
    uploaded_file_paths = []
    errors = []
    if destination == "local":
        async with upload_semaphore:
            for f in file:
                if not f:
                    continue

                safe_filename = os.path.basename(f.filename or "")
                upload_path = upload_dir / safe_filename
                absolute_path = upload_path.absolute()

                # Save file
                if await save_upload_file(f, upload_path):
                    uploaded_filenames.append(safe_filename)
                    uploaded_file_paths.append(str(absolute_path))
                    logger.info(f"Successfully saved file: {safe_filename}")
                else:
                    errors.append(f"Failed to save file: {f.filename}")

    elif destination == "minio":
        minio_results = await upload_to_minio(files=file, folder=folder)
        for result in minio_results:
            if result.get("success"):
                uploaded_filenames.append(result.get("file_name"))
                uploaded_file_paths.append(result.get("object_name"))
            else:
                file_name = result.get('file_name')
                error_msg = result.get('error', 'Unknown error')
                errors.append(f"Failed to upload {file_name}: {error_msg}")

        # Resolve filename conflicts against existing KB documents by renaming (e.g., name -> name_1)
        if index_name:
            try:
                vdb_core = get_vector_db_core()
                existing = await ElasticSearchService.list_files(index_name, include_chunks=False, vdb_core=vdb_core)
                existing_files = existing.get(
                    "files", []) if isinstance(existing, dict) else []
                # Prefer 'file' field; fall back to 'filename' if present
                existing_names = set()
                for item in existing_files:
                    name = (item.get("file") or item.get(
                        "filename") or "").strip()
                    if name:
                        existing_names.add(name.lower())

                def make_unique_names(original_names: List[str], taken_lower: set) -> List[str]:
                    unique_list: List[str] = []
                    local_taken = set(taken_lower)
                    for original in original_names:
                        base, ext = os.path.splitext(original or "")
                        candidate = original or ""
                        if not candidate:
                            unique_list.append(candidate)
                            continue
                        suffix = 1
                        # Ensure case-insensitive uniqueness
                        while candidate.lower() in local_taken:
                            candidate = f"{base}_{suffix}{ext}"
                            suffix += 1
                        unique_list.append(candidate)
                        local_taken.add(candidate.lower())
                    return unique_list

                uploaded_filenames[:] = make_unique_names(
                    uploaded_filenames, existing_names)
            except Exception as e:
                logger.warning(
                    f"Failed to resolve filename conflicts for index '{index_name}': {str(e)}")
    else:
        raise Exception("Invalid destination. Must be 'local' or 'minio'.")
    return errors, uploaded_file_paths, uploaded_filenames


async def upload_to_minio(files: List[UploadFile], folder: str) -> List[dict]:
    """Helper function to upload files to MinIO and return results."""
    results = []
    for f in files:
        try:
            # Read file content
            file_content = await f.read()

            # Convert file content to BytesIO object
            file_obj = BytesIO(file_content)

            # Upload file
            result = upload_fileobj(
                file_obj=file_obj,
                file_name=f.filename or "",
                prefix=folder
            )

            # Reset file pointer for potential re-reading
            await f.seek(0)
            results.append(result)

        except Exception as e:
            # Log single file upload failure but continue processing other files
            logger.error(
                f"Failed to upload file {f.filename}: {e}", exc_info=True)
            results.append({
                "success": False,
                "file_name": f.filename,
                "error": "An error occurred while processing the file."
            })
    return results


async def get_file_url_impl(object_name: str, expires: int):
    result = get_file_url(object_name=object_name, expires=expires)
    if not result["success"]:
        raise Exception(
            f"File does not exist or cannot be accessed: {result.get('error', 'Unknown error')}")
    return result


async def get_file_stream_impl(object_name: str):
    file_stream = get_file_stream(object_name=object_name)
    if file_stream is None:
        raise Exception("File not found or failed to read from storage")
    content_type = get_content_type(object_name)
    return file_stream, content_type


async def delete_file_impl(object_name: str):
    result = delete_file(object_name=object_name)
    if not result["success"]:
        raise Exception(
            f"File does not exist or deletion failed: {result.get('error', 'Unknown error')}")
    return result


async def list_files_impl(prefix: str, limit: Optional[int] = None):
    files = list_files(prefix=prefix)
    if limit:
        files = files[:limit]
    return files


def get_llm_model(tenant_id: str):
    # Get the tenant config
    main_model_config = tenant_config_manager.get_model_config(
        key=MODEL_CONFIG_MAPPING["llm"], tenant_id=tenant_id)
    long_text_to_text_model = OpenAILongContextModel(
        observer=MessageObserver(),
        model_id=get_model_name_from_config(main_model_config),
        api_base=main_model_config.get("base_url"),
        api_key=main_model_config.get("api_key"),
        max_context_tokens=main_model_config.get("max_tokens")
    )
    return long_text_to_text_model

async def process_image_file(query: str, filename: str, file_content: bytes, tenant_id: str, language: str = LANGUAGE["ZH"]) -> str:
    """
    Process image file, convert to text using external API
    """
    # Load messages based on language
    messages = get_file_processing_messages_template(language)
    
    try:
        from utils.attachment_utils import convert_image_to_text
        image_stream = BytesIO(file_content)
        text = convert_image_to_text(query, image_stream, tenant_id, language)
        return messages["IMAGE_CONTENT_SUCCESS"].format(filename=filename, content=text)
    except Exception as e:
        return messages["IMAGE_CONTENT_ERROR"].format(filename=filename, error=str(e))
    
async def process_text_file(query: str, filename: str, file_content: bytes, tenant_id: str, language: str = LANGUAGE["ZH"]) -> tuple[str, Optional[str]]:
    """
    Process text file, convert to text using external API
    """
    # Load messages based on language
    messages = get_file_processing_messages_template(language)
    
    # file_content is byte data, need to send to API through file upload
    from consts.const import DATA_PROCESS_SERVICE
    data_process_service_url = DATA_PROCESS_SERVICE
    api_url = f"{data_process_service_url}/tasks/process_text_file"
    logger.info(f"Processing text file {filename} with API: {api_url}")

    try:
        # Upload byte data as a file
        from utils.file_management_utils import process_text_file as util_process_text_file
        files = {
            'file': (filename, file_content, 'application/octet-stream')
        }
        data = {
            'chunking_strategy': 'basic',
            'timeout': 60
        }
        
        result = await util_process_text_file(api_url, files, data)
        return result.get("response_text", ""), result.get("truncation_percentage")
    except Exception as e:
        logger.error(f"Text file processing error for {filename}: {str(e)}")
        return messages["FILE_CONTENT_ERROR"].format(filename=filename, error=str(e)), None


async def preprocess_files_generator(
    query: str,
    file_cache: List[dict],
    tenant_id: str,
    language: str,
    task_id: str,
    conversation_id: int
) -> AsyncGenerator[str, None]:
    """
    Generate streaming response for file preprocessing

    Args:
        query: User query
        file_cache: List of cached file data
        tenant_id: Tenant ID
        language: Language code
        task_id: Unique task ID
        conversation_id: Conversation ID

    Yields:
        JSON-formatted strings with preprocessing progress and results
    """
    try:
        # Register task
        preprocess_manager.register_preprocess_task(task_id, conversation_id, asyncio.current_task())
        
        # Load messages based on language
        messages = get_file_processing_messages_template(language)
        
        # Check if there are knowledge base tools available
        from agents.create_agent_info import create_tool_config_list
        tool_list = await create_tool_config_list("temp_agent_id", tenant_id, "temp_user_id")
        has_knowledge_base_tool = any(tool.class_name == "KnowledgeBaseSearchTool" for tool in tool_list)
        
        # Initialize final query with original query
        final_query = query
        
        # Process each file
        for file_info in file_cache:
            # Check if task was cancelled
            if asyncio.current_task() and asyncio.current_task().done():
                break
                
            filename = file_info["filename"]
            ext = file_info["ext"]
            
            # Handle file errors
            if "error" in file_info:
                error_msg = messages["FILE_CONTENT_ERROR"].format(
                    filename=filename, error=file_info["error"])
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                continue
                
            file_content = file_info["content"]
            
            try:
                # Determine file type and process accordingly
                if ext in [".txt", ".md", ".csv", ".log", ".json", ".xml"]:
                    # Process text files
                    text_content, truncation_percentage = await process_text_file(
                        query, filename, file_content, tenant_id, language)
                    
                    # Add truncation warning if needed
                    if truncation_percentage:
                        text_content += f"\n\n注意: 文件内容因长度限制被截断 ({truncation_percentage}% 内容被保留)"
                        
                    final_query += f"\n\n{text_content}"
                    yield f"data: {json.dumps({'type': 'file_processed', 'filename': filename, 'description': text_content})}\n\n"
                    
                elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
                    # Process image files
                    image_description = await process_image_file(
                        query, filename, file_content, tenant_id, language)
                    final_query += f"\n\n{image_description}"
                    yield f"data: {json.dumps({'type': 'file_processed', 'filename': filename, 'description': image_description})}\n\n"
                    
                else:
                    # Unsupported file type
                    unsupported_msg = f"不支持的文件类型: {filename}"
                    yield f"data: {json.dumps({'type': 'warning', 'message': unsupported_msg})}\n\n"
                    
            except Exception as e:
                error_msg = messages["FILE_CONTENT_ERROR"].format(
                    filename=filename, error=str(e))
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                
        # 如果有知识库工具，提示用户文档已可用于检索
        if has_knowledge_base_tool:
            final_query += "\n\n注意：上传的文档已可用于知识库检索，您可以在后续对话中引用这些文档内容。"
            yield f"data: {json.dumps({'type': 'info', 'message': '上传的文档已可用于知识库检索'})}\n\n"
                
        # Send completion message
        yield f"data: {json.dumps({'type': 'complete', 'final_query': final_query})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in preprocess_files_generator: {str(e)}")
        error_msg = f"文件预处理过程中发生错误: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    finally:
        # Unregister task
        if task_id in preprocess_manager.tasks:
            del preprocess_manager.tasks[task_id]