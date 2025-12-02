import json
import logging
from typing import List, Optional, Dict, Any
import os
import requests
from pathlib import Path

from pydantic import Field
from smolagents.tools import Tool

from ...vector_database.base import VectorDatabaseCore
from ..models.embedding_model import BaseEmbedding
from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import SearchResultTextMessage, ToolCategory, ToolSign

# Get logger instance
logger = logging.getLogger("knowledge_base_extension_tool")


class KnowledgeBaseExtensionTool(Tool):
    """Enhanced knowledge base tool with extended capabilities for pathology domain"""

    name = "knowledge_base_extension"
    description = (
        "An enhanced knowledge base tool that provides advanced search capabilities for pathology-related documents. "
        "This tool can upload, manage, and search pathology documents including textbooks, research papers, "
        "case studies, and medical images. It supports multimodal search and specialized chunking strategies "
        "suitable for medical content."
    )
    inputs = {
        "operation": {
            "type": "string",
            "description": "Operation to perform. Options: search, upload, list_documents, delete_document",
        },
        "query": {
            "type": "string",
            "description": "Search query when operation is 'search'",
            "nullable": True,
        },
        "search_mode": {
            "type": "string",
            "description": "Search mode: hybrid (default), accurate, semantic",
            "default": "hybrid",
            "nullable": True,
        },
        "file_path": {
            "type": "string",
            "description": "Path to file when operation is 'upload'",
            "nullable": True,
        },
        "document_id": {
            "type": "string",
            "description": "Document ID when operation is 'delete_document'",
            "nullable": True,
        },
        "index_names": {
            "type": "array",
            "description": "List of knowledge base index names to operate on",
            "nullable": True,
        },
        "chunking_strategy": {
            "type": "string",
            "description": "Strategy for chunking documents: basic, by_title, none",
            "default": "by_title",
            "nullable": True,
        },
    }
    output_type = "string"
    category = ToolCategory.SEARCH.value
    tool_sign = ToolSign.KNOWLEDGE_BASE.value

    def __init__(
        self,
        top_k: int = Field(description="Maximum number of search results", default=5),
        index_names: List[str] = Field(description="The list of index names to search", default=None, exclude=True),
        observer: MessageObserver = Field(description="Message observer", default=None, exclude=True),
        embedding_model: BaseEmbedding = Field(description="The embedding model to use", default=None, exclude=True),
        vdb_core: VectorDatabaseCore = Field(description="Vector database client", default=None, exclude=True),
        data_process_service_url: str = Field(description="Data processing service URL", default="http://localhost:5012", exclude=True),
    ):
        """Initialize the enhanced knowledge base tool.

        Args:
            top_k (int, optional): Number of results to return. Defaults to 5.
            index_names (List[str], optional): List of index names to operate on.
            observer (MessageObserver, optional): Message observer instance.
            embedding_model (BaseEmbedding, optional): Embedding model to use.
            vdb_core (VectorDatabaseCore, optional): Vector database client.
            data_process_service_url (str, optional): URL of the data processing service.
        """
        super().__init__()
        self.top_k = top_k
        self.index_names = index_names if index_names is not None else []
        self.observer = observer
        self.embedding_model = embedding_model
        self.vdb_core = vdb_core
        self.data_process_service_url = data_process_service_url
        self.running_prompt_zh = "知识库操作处理中..."
        self.running_prompt_en = "Processing knowledge base operation..."

    def _send_tool_message(self, operation: str):
        """Send tool running message to observer"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "database", "text": f"Knowledge Base {operation}"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def search(self, query: str, search_mode: str = "hybrid", index_names: Optional[List[str]] = None) -> str:
        """Perform search in knowledge base"""
        self._send_tool_message("search")
        
        search_index_names = index_names if index_names is not None else self.index_names
        
        logger.info(
            f"KnowledgeBaseExtensionTool search called with query: '{query}', search_mode: '{search_mode}', index_names: {search_index_names}"
        )
        
        if len(search_index_names) == 0:
            return json.dumps("No knowledge base selected. No relevant information found.", ensure_ascii=False)

        try:
            if search_mode == "hybrid":
                kb_search_data = self._search_hybrid(query=query, index_names=search_index_names)
            elif search_mode == "accurate":
                kb_search_data = self._search_accurate(query=query, index_names=search_index_names)
            elif search_mode == "semantic":
                kb_search_data = self._search_semantic(query=query, index_names=search_index_names)
            else:
                raise Exception(f"Invalid search mode: {search_mode}, only support: hybrid, accurate, semantic")

            kb_search_results = kb_search_data["results"]

            if not kb_search_results:
                raise Exception("No results found! Try a less restrictive/shorter query.")

            search_results_json = []
            for index, single_search_result in enumerate(kb_search_results):
                source_type = single_search_result.get("source_type", "")
                source_type = "file" if source_type in ["local", "minio"] else source_type
                title = single_search_result.get("title")
                if not title:
                    title = single_search_result.get("filename", "")
                    
                search_result_message = SearchResultTextMessage(
                    title=title,
                    text=single_search_result.get("content", ""),
                    source_type=source_type,
                    url=single_search_result.get("path_or_url", ""),
                    filename=single_search_result.get("filename", ""),
                    published_date=single_search_result.get("create_time", ""),
                    score=single_search_result.get("score", 0),
                    score_details=single_search_result.get("score_details", {}),
                    cite_index=index,
                    search_type=self.name,
                    tool_sign=self.tool_sign,
                )
                
                search_results_json.append(search_result_message.to_dict())
                
            return json.dumps(search_results_json, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error during knowledge base search: {str(e)}")
            raise e

    def upload_document(
        self, 
        file_path: str, 
        index_names: Optional[List[str]] = None, 
        chunking_strategy: str = "by_title"
    ) -> str:
        """Upload a document to the knowledge base"""
        self._send_tool_message("upload")
        
        target_index_names = index_names if index_names is not None else self.index_names
        
        logger.info(
            f"KnowledgeBaseExtensionTool upload called with file_path: '{file_path}', index_names: {target_index_names}, chunking_strategy: {chunking_strategy}"
        )
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file name
            file_name = os.path.basename(file_path)
            
            # Process each index separately
            results = []
            for index_name in target_index_names:
                # Prepare the request to data processing service
                url = f"{self.data_process_service_url}/tasks"
                
                # Prepare payload
                payload = {
                    "source": file_path,
                    "source_type": "local",
                    "chunking_strategy": chunking_strategy,
                    "index_name": index_name,
                    "original_filename": file_name
                }
                
                # Make request to data processing service
                try:
                    response = requests.post(url, json=payload, timeout=30)
                    if response.status_code == 201:
                        task_data = response.json()
                        results.append({
                            "index": index_name,
                            "status": "success",
                            "task_id": task_data.get("task_id"),
                            "message": f"Document queued for processing in index {index_name}"
                        })
                    else:
                        results.append({
                            "index": index_name,
                            "status": "error",
                            "message": f"Failed to queue document processing for index {index_name}: {response.text}"
                        })
                except requests.RequestException as e:
                    results.append({
                        "index": index_name,
                        "status": "error",
                        "message": f"Failed to connect to data processing service for index {index_name}: {str(e)}"
                    })
            
            result = {
                "status": "completed",
                "file_path": file_path,
                "indexes": target_index_names,
                "chunking_strategy": chunking_strategy,
                "results": results,
                "message": f"Document processing initiated for {len(target_index_names)} index(es)"
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error during document upload: {str(e)}")
            raise e

    def list_documents(self, index_names: Optional[List[str]] = None) -> str:
        """List documents in the knowledge base"""
        self._send_tool_message("list_documents")
        
        target_index_names = index_names if index_names is not None else self.index_names
        
        logger.info(f"KnowledgeBaseExtensionTool list_documents called with index_names: {target_index_names}")
        
        try:
            all_documents = []
            
            # Get documents for each index
            for index_name in target_index_names:
                try:
                    # Get document details from vector database
                    documents = self.vdb_core.get_documents_detail(index_name)
                    for doc in documents:
                        doc["index_name"] = index_name
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Failed to list documents for index {index_name}: {str(e)}")
                    # Add error entry for this index
                    all_documents.append({
                        "index_name": index_name,
                        "error": str(e)
                    })
            
            result = {
                "status": "success",
                "operation": "list_documents",
                "indexes": target_index_names,
                "message": f"Retrieved document list for {len(target_index_names)} index(es)",
                "documents": all_documents
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error during document listing: {str(e)}")
            raise e

    def delete_document(self, document_id: str, index_names: Optional[List[str]] = None) -> str:
        """Delete a document from the knowledge base"""
        self._send_tool_message("delete_document")
        
        target_index_names = index_names if index_names is not None else self.index_names
        
        logger.info(
            f"KnowledgeBaseExtensionTool delete_document called with document_id: '{document_id}', index_names: {target_index_names}"
        )
        
        try:
            results = []
            
            # Delete document from each index
            for index_name in target_index_names:
                try:
                    # Delete documents from vector database
                    deleted_count = self.vdb_core.delete_documents(index_name, document_id)
                    results.append({
                        "index": index_name,
                        "status": "success",
                        "deleted_count": deleted_count,
                        "message": f"Deleted {deleted_count} documents from index {index_name}"
                    })
                except Exception as e:
                    logger.error(f"Failed to delete document from index {index_name}: {str(e)}")
                    results.append({
                        "index": index_name,
                        "status": "error",
                        "message": str(e)
                    })
            
            result = {
                "status": "completed",
                "document_id": document_id,
                "indexes": target_index_names,
                "results": results,
                "message": f"Document deletion processed for {len(target_index_names)} index(es)"
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error during document deletion: {str(e)}")
            raise e

    def _search_hybrid(self, query: str, index_names: List[str]) -> Dict[str, Any]:
        """Perform hybrid search combining accurate and semantic search"""
        try:
            results = self.vdb_core.hybrid_search(
                index_names=index_names,
                query_text=query,
                embedding_model=self.embedding_model,
                top_k=self.top_k
            )
            return {"total": len(results), "results": results}
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            raise Exception(f"Error during hybrid search: {str(e)}")

    def _search_accurate(self, query: str, index_names: List[str]) -> Dict[str, Any]:
        """Perform accurate text matching search"""
        try:
            results = self.vdb_core.accurate_search(
                index_names=index_names,
                query_text=query,
                top_k=self.top_k
            )
            return {"total": len(results), "results": results}
        except Exception as e:
            logger.error(f"Error during accurate search: {str(e)}")
            raise Exception(f"Error during accurate search: {str(e)}")

    def _search_semantic(self, query: str, index_names: List[str]) -> Dict[str, Any]:
        """Perform semantic similarity search"""
        try:
            results = self.vdb_core.semantic_search(
                index_names=index_names,
                query_text=query,
                embedding_model=self.embedding_model,
                top_k=self.top_k
            )
            return {"total": len(results), "results": results}
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            raise Exception(f"Error during semantic search: {str(e)}")

    def forward(
        self, 
        operation: str,
        query: Optional[str] = None,
        search_mode: str = "hybrid",
        file_path: Optional[str] = None,
        document_id: Optional[str] = None,
        index_names: Optional[List[str]] = None,
        chunking_strategy: str = "by_title"
    ) -> str:
        """Main entry point for the tool operations"""
        try:
            if operation == "search":
                if not query:
                    raise ValueError("Query is required for search operation")
                return self.search(query, search_mode, index_names)
                
            elif operation == "upload":
                if not file_path:
                    raise ValueError("File path is required for upload operation")
                return self.upload_document(file_path, index_names, chunking_strategy)
                
            elif operation == "list_documents":
                return self.list_documents(index_names)
                
            elif operation == "delete_document":
                if not document_id:
                    raise ValueError("Document ID is required for delete operation")
                return self.delete_document(document_id, index_names)
                
            else:
                raise ValueError(f"Unsupported operation: {operation}. Supported operations: search, upload, list_documents, delete_document")
                
        except Exception as e:
            logger.error(f"Error in KnowledgeBaseExtensionTool: {str(e)}")
            raise e