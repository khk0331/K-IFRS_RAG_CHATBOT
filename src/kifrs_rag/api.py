import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    )
    return response


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query")
def query(payload: QueryRequest) -> dict:
    try:
        return get_service().query(payload.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
