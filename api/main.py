"""FastAPI app for the AI Lawyer assistant.

Wraps app/rag.py::answer_question() in a thin HTTP layer so the Streamlit
frontend (or any other client) can talk to RAG over /ask.

Run from the project root:
    uvicorn api.main:app --reload --port 8000
"""
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from api.routes.ask import router as ask_router
from api.routes.health import router as health_router

app = FastAPI(
    title="AI Lawyer API",
    version="1.0.0",
    description="GraphRAG over Egyptian Labor Law No. 14 of 2025",
)

app.include_router(ask_router)
app.include_router(health_router)


@app.get("/")
def home():
    return {"message": "AI Lawyer API Running"}
