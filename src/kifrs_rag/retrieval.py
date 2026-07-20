import math
import re
from collections import Counter

from .models import Chunk, SearchResult


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(text):
        token = raw.lower()
        if len(token) <= 1:
            continue
        tokens.append(token)
        if re.fullmatch(r"[가-힣]+", token):
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    common = left.keys() & right.keys()
    numerator = sum(left[token] * right[token] for token in common)
    denominator = math.sqrt(sum(v * v for v in left.values())) * math.sqrt(
        sum(v * v for v in right.values())
    )
    return numerator / denominator if denominator else 0.0


class LocalRetriever:
    """Deterministic local baseline; replaceable with a production embedding provider."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._vectors = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        scored = [
            SearchResult(chunk=chunk, score=score)
            for chunk, score in zip(self._chunks, self.score_all(question), strict=True)
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    def score_all(self, question: str) -> list[float]:
        query = Counter(tokenize(question))
        return [_cosine(query, vector) for vector in self._vectors]


class DemoHybridRetriever:
    """Dependency-free hybrid retriever for the public synthetic demo."""

    def __init__(self, chunks: list[Chunk]):
        from .hybrid_retrieval import BM25Retriever

        self._chunks = chunks
        self._vector = LocalRetriever(chunks)
        self._bm25 = BM25Retriever(chunks)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        vector_scores = self._vector.score_all(question)
        keyword_scores = self._bm25.scores(question)
        max_keyword = max(keyword_scores, default=1.0) or 1.0
        question_tokens = set(tokenize(question))
        results: list[SearchResult] = []
        for index, chunk in enumerate(self._chunks):
            title_overlap = len(question_tokens & set(tokenize(chunk.title)))
            rerank_bonus = min(title_overlap * 0.035, 0.14)
            score = (
                0.58 * vector_scores[index]
                + 0.32 * keyword_scores[index] / max_keyword
                + rerank_bonus
            )
            results.append(SearchResult(chunk=chunk, score=min(score, 1.0)))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
