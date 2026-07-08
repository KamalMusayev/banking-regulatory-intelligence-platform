"""
app/api/chat.py

API routes for the chat service.
Keep routers extremely thin, delegating all mapping and processing logic to ChatService.
"""
from __future__ import annotations

from typing import Any

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends

from app.core.dependencies import get_generation_pipeline
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Send a question to the RAG pipeline")
def chat(
    request: ChatRequest,
    pipeline: Any = Depends(get_generation_pipeline),
) -> ChatResponse:
    """
    Submit a question in Azerbaijani to the banking regulations assistant.
    
    The request is validated, passed to the ChatService for execution in the
    GenerationPipeline, and formatted to return inline citations and retrieval metrics.
    """
    service = ChatService(pipeline)
    return service.process_query(request)
