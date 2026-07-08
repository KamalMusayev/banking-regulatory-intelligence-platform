"""
app/services/chat_service.py

Service layer for Chat functionality.
Acts as a buffer between the API controller/router and the core AI RAG pipeline.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument, MetricsResponse
from backend.reguaz.utils.logger import get_logger

# Initialize project-wide logger for API requests
logger = get_logger("app.api.requests", "api_requests.log")


class ChatService:
    """
    Service responsible for orchestrating chat queries and mapping
    the output of the GenerationPipeline to clean API schemas.
    """

    def __init__(self, generation_pipeline: Any) -> None:
        """
        Initialize the ChatService with an instance of the GenerationPipeline.
        
        Args:
            generation_pipeline: The backend RAG generation pipeline instance.
        """
        self.generation_pipeline = generation_pipeline

    def process_query(self, request: ChatRequest) -> ChatResponse:
        """
        Executes the query through the RAG generation pipeline and formats the response.

        Args:
            request: The validated ChatRequest containing the user's question.

        Returns:
            A formatted ChatResponse.
        """
        try:
            # Call the existing GenerationPipeline.generate method
            # The pipeline returns:
            # {
            #     "question": str,
            #     "answer": str,
            #     "sources": list[dict],
            #     "metrics": {
            #         "retrieval_time": float,
            #         "prompt_build_time": float,
            #         "generation_time": float,
            #         "total_time": float
            #     }
            # }
            pipeline_output: Dict[str, Any] = self.generation_pipeline.generate(request.question)

            # Map metrics
            metrics_data = pipeline_output.get("metrics", {})
            retrieval_time = metrics_data.get("retrieval_time", 0.0)
            generation_time = metrics_data.get("generation_time", 0.0)
            total_time = metrics_data.get("total_time", 0.0)

            metrics = MetricsResponse(
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time
            )

            # Log request success details
            logger.info(
                "CHAT_REQUEST SUCCESS | question='%s' | retrieval_time=%.3fs | generation_time=%.3fs | total_time=%.3fs",
                request.question,
                retrieval_time,
                generation_time,
                total_time
            )

        except Exception as exc:
            # Log request failure details
            logger.error(
                "CHAT_REQUEST FAILURE | question='%s' | error='%s'",
                request.question,
                str(exc),
                exc_info=True
            )
            raise

        # Map sources and construct citation indices
        sources = []
        for idx, source_dict in enumerate(pipeline_output.get("sources", []), start=1):
            text_content = source_dict.get("text") or source_dict.get("content") or ""
            preview = text_content[:300]

            source_doc = SourceDocument(
                citation=idx,
                chunk_id=source_dict.get("chunk_id") or source_dict.get("id") or "",
                document_id=source_dict.get("document_id"),
                document_name=source_dict.get("title") or source_dict.get("document_name") or "Unknown Document",
                category=source_dict.get("category", "unknown"),
                chapter=source_dict.get("chapter"),
                article=source_dict.get("article"),
                page=source_dict.get("page_start"),
                chunk_preview=preview,
                # Optional scoring metadata if present in pipeline output
                rerank_score=source_dict.get("rerank_score"),
                rrf_score=source_dict.get("rrf_score"),
                semantic_rank=source_dict.get("semantic_rank"),
                bm25_rank=source_dict.get("bm25_rank")
            )
            sources.append(source_doc)

        # Generate or reuse session id
        resp_session_id = request.session_id or str(uuid.uuid4())

        return ChatResponse(
            session_id=resp_session_id,
            question=request.question,
            answer=pipeline_output.get("answer", ""),
            sources=sources,
            metrics=metrics
        )
