from pydantic import BaseModel
from typing import List, Dict


class QuestionRequest(BaseModel):
    question: str


class ChunkResponse(BaseModel):
    node_id: str
    node_type: str
    article_number: int | None
    text: str
    score: float
    source: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    chunks: List[ChunkResponse]
    elapsed_ms: Dict[str, int]