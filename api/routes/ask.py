from fastapi import APIRouter

from api.schemas.chat import (
    QuestionRequest,
    QuestionResponse,
)

router = APIRouter()


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    question = request.question

    return QuestionResponse(
        answer=f"You asked: {question}"
    )