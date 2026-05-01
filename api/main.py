from fastapi import FastAPI
from api.routes.ask import router as ask_router

app = FastAPI(
    title="AI Lawyer API",
    version="1.0.0"
)

app.include_router(ask_router)


@app.get("/")
def home():
    return {
        "message": "AI Lawyer API Running"
    }