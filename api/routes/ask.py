from fastapi import APIRouter

from api.schemas.chat import (
    QuestionRequest,
    QuestionResponse,
)

from api.services.rag_service import ask_rag

router = APIRouter()


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    answer = ask_rag(request.question)

    return QuestionResponse(
        answer=answer
    )