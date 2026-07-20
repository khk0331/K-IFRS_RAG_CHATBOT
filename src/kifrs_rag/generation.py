from typing import Protocol

from .models import SearchResult


class Generator(Protocol):
    def generate(self, question: str, evidence: list[SearchResult]) -> str: ...


class ExtractiveGenerator:
    """Safe offline baseline that copies the highest-ranked evidence."""

    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        del question
        lead = evidence[0].chunk
        return f"{lead.text} ({lead.standard_id} 문단 {lead.paragraph_id})"

