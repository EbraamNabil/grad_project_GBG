from fastapi import APIRouter

from api.schemas.chat import (
    QuestionRequest,
    QuestionResponse,
    ChunkResponse
)

from api.services.rag_service import ask_rag

router = APIRouter()


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    result = ask_rag(request.question)

    return QuestionResponse(
        question=result.question,
        answer=result.answer,

        chunks=[
            ChunkResponse(
                node_id=chunk.node_id,
                node_type=chunk.node_type,
                article_number=chunk.article_number,
                text=chunk.text,
                score=chunk.score,
                source=chunk.source
            )
            for chunk in result.chunks
        ],

        elapsed_ms=result.elapsed_ms
    )