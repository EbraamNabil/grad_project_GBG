"""Request/response schemas for /ask.

Expanded from coworker's original to expose all the fields the Streamlit UI
needs: source-type label, node_type, breadcrumb, full text, detected_refs.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str
    primary_k: Optional[int] = Field(
        default=None,
        description="Override PRIMARY_K from app/rag.py for this query.",
    )


class SourceResponse(BaseModel):
    node_id: str
    node_type: str           # "Article" | "ArticleSegment" | "Definition"
    article: Optional[int]
    breadcrumb: str
    excerpt: str             # first 300 chars (for collapsed display)
    text: str                # full text (for expansion / keyword highlighting)
    score: float
    source: str              # "explicit" | "primary" | "definition" | "cross_ref"


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]
    detected_refs: List[int] = Field(default_factory=list)
    elapsed_ms: Dict[str, int]
