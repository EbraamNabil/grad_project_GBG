"""Thin wrapper around app/rag.py::answer_question.

Lives in services/ so future cross-cutting concerns (caching, logging,
rate-limiting) can be added here without polluting the route handler.
"""
from app.rag import DEFAULT_MODE, answer_question, RagResponse


def ask_rag(
    question: str,
    mode: str = DEFAULT_MODE,
    primary_k: int | None = None,
) -> RagResponse:
    return answer_question(question, primary_k=primary_k, mode=mode)
