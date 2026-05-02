from pydantic import BaseModel
from typing import List, Dict


class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    article: int | None
    excerpt: str
    score: float


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]
    elapsed_ms: Dict[str, int] # we used elapsed_ms to provide timing information about the RAG process, which can be useful for performance monitoring and debugging.