from uuid import uuid4

from .generation import (
    ExtractiveGenerator,
    GenerationError,
    Generator,
    validate_generation,
)
from .guardrails import citations_are_valid, sufficient_evidence, validate_question
from typing import Protocol

from .models import SearchResult


class Retriever(Protocol):
    def search(self, question: str, top_k: int = 3) -> list[SearchResult]: ...


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        min_score: float = 0.18,
        top_k: int = 3,
        generator: Generator | None = None,
    ):
        self.retriever = retriever
        self.min_score = min_score
        self.top_k = top_k
        self.generator = generator or ExtractiveGenerator()

    def query(self, question: str) -> dict:
        clean = validate_question(question)
        results = self.retriever.search(clean, self.top_k)
        trace_id = str(uuid4())
        if not sufficient_evidence(results, self.min_score):
            return {
                "status": "insufficient_evidence",
                "answer": None,
                "citations": [],
                "trace_id": trace_id,
            }

        grounded = [result for result in results if result.score >= self.min_score]
        if not citations_are_valid(grounded):
            return {"status": "validation_failed", "answer": None, "citations": [], "trace_id": trace_id}

        try:
            generation = self.generator.generate(clean, grounded)
            validate_generation(generation, len(grounded))
        except GenerationError:
            return {
                "status": "validation_failed",
                "answer": None,
                "citations": [],
                "trace_id": trace_id,
            }
        cited_results = [grounded[index] for index in generation.evidence_indices]
        citations = [
            {
                "standard_id": result.chunk.standard_id,
                "paragraph_id": result.chunk.paragraph_id,
                "quote": result.chunk.text,
                "score": round(result.score, 4),
                "source": result.chunk.source,
            }
            for result in cited_results
        ]
        return {
            "status": "answered",
            "answer": generation.answer,
            "citations": citations,
            "trace_id": trace_id,
        }
