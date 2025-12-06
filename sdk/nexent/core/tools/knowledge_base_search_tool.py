import json
import logging
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import Field
from smolagents.tools import Tool

from ...vector_database.base import VectorDatabaseCore
from ..models.embedding_model import BaseEmbedding
from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import SearchResultTextMessage, ToolCategory, ToolSign


# Get logger instance
logger = logging.getLogger("knowledge_base_search_tool")


class KnowledgeBaseSearchTool(Tool):
    """Knowledge base search tool"""

    name = "knowledge_base_search"
    description = (
        "Performs a local knowledge base search based on your query then returns the top search results. "
        "A tool for retrieving domain-specific knowledge, documents, and information stored in the local knowledge base. "
        "Use this tool when users ask questions related to specialized knowledge, technical documentation, "
        "domain expertise, personal notes, or any information that has been indexed in the knowledge base. "
        "Suitable for queries requiring access to stored knowledge that may not be publicly available."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "search_mode": {
            "type": "string",
            "description": "the search mode, optional values: hybrid, combining accurate matching and semantic search results across multiple indices.; accurate, Search for documents using fuzzy text matching across multiple indices; semantic, Search for similar documents using vector similarity across multiple indices.",
            "default": "hybrid",
            "nullable": True,
        },
        "index_names": {
            "type": "array",
            "description": "The list of knowledge base index names to search. If not provided, will search all available knowledge bases.",
            "nullable": True,
        },
    }
    output_type = "string"
    category = ToolCategory.SEARCH.value

    # Used to distinguish different index sources for summaries
    tool_sign = ToolSign.KNOWLEDGE_BASE.value

    def __init__(
        self,
        top_k: int = 5,
        index_names: Optional[List[str]] = None,
        observer: Optional[MessageObserver] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        vdb_core: Optional[VectorDatabaseCore] = None,
        max_workers: int = 4,
    ):
        """Initialize the KBSearchTool.

        Args:
            top_k (int, optional): Number of results to return. Defaults to 5.
            index_names (List[str], optional): The list of index names to search. Defaults to None.
            observer (MessageObserver, optional): Message observer instance. Defaults to None.
            embedding_model (BaseEmbedding, optional): The embedding model to use. Defaults to None.
            vdb_core (VectorDatabaseCore, optional): Vector database client. Defaults to None.
            max_workers (int, optional): Maximum number of worker threads for parallel processing. Defaults to 4.

        Raises:
            ValueError: If language is not supported
        """
        super().__init__()
        # Ensure parameters are properly handled
        self.top_k = top_k if isinstance(top_k, int) else 5
        self.observer = observer
        self.vdb_core = vdb_core
        self.index_names = [] if index_names is None else index_names
        self.embedding_model = embedding_model
        self.max_workers = max_workers if isinstance(max_workers, int) else 4

        self.record_ops = 1  # To record serial number
        self.running_prompt_zh = "知识库检索中..."
        self.running_prompt_en = "Searching the knowledge base..."

    def _search_single_index(self, query: str, search_mode: str, index_name: str) -> dict:
        """Search in a single index and return results."""
        try:
            if search_mode == "hybrid":
                results = self.vdb_core.hybrid_search(
                    index_names=[index_name], query_text=query, embedding_model=self.embedding_model, top_k=self.top_k
                )
            elif search_mode == "accurate":
                results = self.vdb_core.accurate_search(index_names=[index_name], query_text=query, top_k=self.top_k)
            elif search_mode == "semantic":
                results = self.vdb_core.semantic_search(
                    index_names=[index_name], query_text=query, embedding_model=self.embedding_model, top_k=self.top_k
                )
            else:
                raise ValueError(f"Invalid search mode: {search_mode}")

            # Format results
            formatted_results = []
            for result in results:
                doc = result["document"]
                doc["score"] = result["score"]
                # Include source index in results
                doc["index"] = result["index"]
                formatted_results.append(doc)

            return {
                "index": index_name,
                "results": formatted_results,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error searching index {index_name}: {str(e)}")
            return {
                "index": index_name,
                "results": [],
                "status": "error",
                "error": str(e)
            }

    def _merge_results(self, search_results: List[dict]) -> List[dict]:
        """Merge results from multiple indices and sort by score."""
        all_results = []
        for result_set in search_results:
            if result_set["status"] == "success":
                all_results.extend(result_set["results"])

        # Sort by score in descending order
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:self.top_k]

    def forward(self, query: str, search_mode: str = "hybrid", index_names: Optional[List[str]] = None) -> str:
        # Send tool run message
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "search", "text": query}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

        # Use provided index_names if available, otherwise use default
        search_index_names = index_names if index_names is not None else self.index_names

        # Log the index_names being used for this search
        logger.info(
            f"KnowledgeBaseSearchTool called with query: '{query}', search_mode: '{search_mode}', index_names: {search_index_names}"
        )

        # Ensure index_names is always a list
        if not isinstance(search_index_names, list):
            search_index_names = []

        if len(search_index_names) == 0:
            return json.dumps("No knowledge base selected. No relevant information found.", ensure_ascii=False)

        # Handle parallel or sequential search based on number of indices
        search_results = []
        if len(search_index_names) > 1:
            # Use parallel processing for multiple indices
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all search tasks
                future_to_index = {
                    executor.submit(self._search_single_index, query, search_mode, index_name): index_name 
                    for index_name in search_index_names
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    result = future.result()
                    search_results.append(result)
        else:
            # Use original method for single index to maintain backward compatibility
            if search_mode == "hybrid":
                kb_search_data = self.search_hybrid(query=query, index_names=search_index_names)
            elif search_mode == "accurate":
                kb_search_data = self.search_accurate(query=query, index_names=search_index_names)
            elif search_mode == "semantic":
                kb_search_data = self.search_semantic(query=query, index_names=search_index_names)
            else:
                raise Exception(f"Invalid search mode: {search_mode}, only support: hybrid, accurate, semantic")

            # Convert to the new format
            search_results = [{
                "index": "unknown",
                "results": kb_search_data["results"],
                "status": "success"
            }]

        # Merge results from all indices
        kb_search_results = self._merge_results(search_results)

        if not kb_search_results:
            raise Exception("No results found! Try a less restrictive/shorter query.")

        search_results_json = []  # Organize search results into a unified format
        search_results_return = []  # Format for input to the large model
        for index, single_search_result in enumerate(kb_search_results):
            # Temporarily correct the source_type stored in the knowledge base
            source_type = single_search_result.get("source_type", "")
            source_type = "file" if source_type in ["local", "minio"] else source_type
            
            # 提取标题，确保至少有一个标题
            title = single_search_result.get("title")
            if not title:
                title = single_search_result.get("filename", "未命名文档")
                
            # 确保URL字段始终存在
            url = single_search_result.get("path_or_url", "")
            if not url:
                url = single_search_result.get("url", "")
                
            search_result_message = SearchResultTextMessage(
                title=title,
                text=single_search_result.get("content", ""),
                source_type=source_type,
                url=url,
                filename=single_search_result.get("filename", ""),
                published_date=single_search_result.get("create_time", ""),
                score=single_search_result.get("score", 0),
                score_details=single_search_result.get("score_details", {}),
                cite_index=self.record_ops + index,
                search_type=self.name,
                tool_sign=self.tool_sign,
            )

            search_results_json.append(search_result_message.to_dict())
            search_results_return.append(search_result_message.to_model_dict())

        self.record_ops += len(search_results_return)

        # Record the detailed content of this search
        if self.observer:
            search_results_data = json.dumps(search_results_json, ensure_ascii=False)
            self.observer.add_message("", ProcessType.SEARCH_CONTENT, search_results_data)
            
        formatted_results = []
        for i, result in enumerate(search_results_return):
            # 生成引用标记，格式为[[字母+数字]]，例如[[a1]], [[b2]]
            alphabet = chr(ord('a') + (self.record_ops - len(search_results_return) + i) // 10)
            number = (self.record_ops - len(search_results_return) + i) % 10 + 1
            citation_mark = f"[[{alphabet}{number}]]"
            
            # 确保所有必需字段都存在
            formatted_result = {
                "content": result.get("text", ""),
                "title": result.get("title", "未命名文档"),
                "url": result.get("url", ""),
                "citation_mark": citation_mark,
                "score": result.get("score", 0)
            }
            formatted_results.append(formatted_result)
            
        return json.dumps(formatted_results, ensure_ascii=False)
        
    def search_hybrid(self, query, index_names):
        try:
            results = self.vdb_core.hybrid_search(
                index_names=index_names, query_text=query, embedding_model=self.embedding_model, top_k=self.top_k
            )

            # Format results
            formatted_results = []
            for result in results:
                doc = result["document"]
                doc["score"] = result["score"]
                # Include source index in results
                doc["index"] = result["index"]
                formatted_results.append(doc)

            return {
                "results": formatted_results,
                "total": len(formatted_results),
            }
        except Exception as e:
            raise Exception(f"Error during semantic search: {str(e)}")

    def search_accurate(self, query, index_names):
        try:
            results = self.vdb_core.accurate_search(index_names=index_names, query_text=query, top_k=self.top_k)

            # Format results
            formatted_results = []
            for result in results:
                doc = result["document"]
                doc["score"] = result["score"]
                # Include source index in results
                doc["index"] = result["index"]
                formatted_results.append(doc)

            return {
                "results": formatted_results,
                "total": len(formatted_results),
            }
        except Exception as e:
            raise Exception(f"Error during accurate search: {str(e)}")

    def search_semantic(self, query, index_names):
        try:
            results = self.vdb_core.semantic_search(
                index_names=index_names, query_text=query, embedding_model=self.embedding_model, top_k=self.top_k
            )

            # Format results
            formatted_results = []
            for result in results:
                doc = result["document"]
                doc["score"] = result["score"]
                # Include source index in results
                doc["index"] = result["index"]
                formatted_results.append(doc)

            return {
                "results": formatted_results,
                "total": len(formatted_results),
            }
        except Exception as e:
            raise Exception(f"Error during semantic search: {str(e)}")