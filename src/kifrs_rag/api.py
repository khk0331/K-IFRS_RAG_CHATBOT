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
    retriever_name = os.getenv("KIFRS_RETRIEVER", "local")
    if retriever_name == "dense":
        from .dense_retrieval import DenseRetriever

        retriever = DenseRetriever(os.getenv("KIFRS_INDEX_PATH", "data/index/e5-small"))
    elif retriever_name == "local":
        retriever = LocalRetriever(load_chunks(path))
    else:
        raise ValueError(f"unsupported retriever: {retriever_name}")
    default_min_score = "0.86" if retriever_name == "dense" else "0.18"
    generator_name = os.getenv("KIFRS_GENERATOR", "extractive")
    if generator_name == "openai_compatible":
        from .generation import OpenAICompatibleGenerator

        generator = OpenAICompatibleGenerator(
            base_url=os.environ["KIFRS_LLM_BASE_URL"],
            model=os.environ["KIFRS_LLM_MODEL"],
            api_key=os.getenv("KIFRS_LLM_API_KEY"),
        )
    elif generator_name == "extractive":
        generator = None
    else:
        raise ValueError(f"unsupported generator: {generator_name}")
    return RagService(
        retriever,
        min_score=float(os.getenv("KIFRS_MIN_SCORE", default_min_score)),
        top_k=int(os.getenv("KIFRS_TOP_K", "3")),
        generator=generator,
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
