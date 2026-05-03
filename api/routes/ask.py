"""POST /ask — main RAG endpoint."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.core.logger import logger
from api.schemas.chat import (
    QuestionRequest,
    QuestionResponse,
    SourceResponse,
)
from api.services.rag_service import ask_rag

router = APIRouter()


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    try:
        logger.info(f"Question received: {request.question}")
        result = ask_rag(request.question, primary_k=request.primary_k)
        logger.info(
            f"RAG response: {len(result.chunks)} chunks, "
            f"detected_refs={result.detected_refs}, "
            f"elapsed={result.elapsed_ms}"
        )

        sources = [
            SourceResponse(
                node_id=chunk.node_id,
                node_type=chunk.node_type,
                article=chunk.article_number,
                breadcrumb=chunk.breadcrumb,
                excerpt=chunk.text[:300],
                text=chunk.text,
                score=round(chunk.score, 3),
                source=chunk.source,
            )
            for chunk in result.chunks
        ]

        return QuestionResponse(
            question=result.question,
            answer=result.answer,
            sources=sources,
            detected_refs=result.detected_refs,
            elapsed_ms=result.elapsed_ms,
        )

    except Exception as e:
        logger.exception(f"Error processing question: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
