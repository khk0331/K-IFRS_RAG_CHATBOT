import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ingestion import load_chunks
from .retrieval import LocalRetriever
from .service import RagService


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@lru_cache
def get_service() -> RagService:
    path = os.getenv("KIFRS_DATA_PATH", "data/sample/standards.json")
    chunks = load_chunks(path)
    return RagService(
        LocalRetriever(chunks),
        min_score=float(os.getenv("KIFRS_MIN_SCORE", "0.18")),
        top_k=int(os.getenv("KIFRS_TOP_K", "3")),
    )


app = FastAPI(title="K-IFRS RAG", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query")
def query(payload: QueryRequest) -> dict:
    try:
        return get_service().query(payload.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

