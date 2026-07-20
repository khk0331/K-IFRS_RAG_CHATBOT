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
        query = Counter(tokenize(question))
        scored = [
            SearchResult(chunk=chunk, score=_cosine(query, vector))
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]
