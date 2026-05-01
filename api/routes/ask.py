from api.core.logger import logger
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas.chat import (
    QuestionRequest,
    QuestionResponse,
    SourceResponse
)

from api.services.rag_service import ask_rag

router = APIRouter()


@router.post("/ask", response_model=QuestionResponse)
   
def ask_question(request: QuestionRequest):
  
    try:
        logger.info(f"Question received: {request.question}") # to log the incoming question
        result = ask_rag(request.question)
        logger.info("RAG response generated successfully")

        sources = []

        for chunk in result.chunks[:3]:

            excerpt = chunk.text[:300]

            sources.append(
                SourceResponse(
                    article=chunk.article_number,
                    excerpt=excerpt,
                    score=round(chunk.score, 3)
                )
            )

        return QuestionResponse(
            question=result.question,
            answer=result.answer,
            sources=sources,
            elapsed_ms=result.elapsed_ms
        )

    except Exception as e:

        logger.error(f"Error occurred while processing question: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )