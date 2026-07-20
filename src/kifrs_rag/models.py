from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    standard_id: str
    paragraph_id: str
    title: str
    effective_date: str
    source: str
    text: str
    chunk_index: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float
