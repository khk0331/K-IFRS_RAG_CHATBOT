import re

from .models import SearchResult


INJECTION_PATTERNS = (
    re.compile(r"이전.{0,10}(지시|명령).{0,5}(무시|잊어)"),
    re.compile(r"ignore.{0,20}(previous|system).{0,20}(instruction|prompt)", re.I),
    re.compile(r"system\s*prompt", re.I),
)


def validate_question(question: str, max_length: int = 1000) -> str:
    clean = " ".join(question.split())
    if not clean:
        raise ValueError("질문을 입력해 주세요.")
    if len(clean) > max_length:
        raise ValueError(f"질문은 {max_length}자 이하여야 합니다.")
    if any(pattern.search(clean) for pattern in INJECTION_PATTERNS):
        raise ValueError("안전하지 않은 지시문이 감지되었습니다.")
    return clean


def sufficient_evidence(results: list[SearchResult], min_score: float) -> bool:
    return bool(results) and results[0].score >= min_score


def citations_are_valid(results: list[SearchResult]) -> bool:
    return all(
        result.chunk.standard_id
        and result.chunk.paragraph_id
        and result.chunk.text
        and 0 <= result.score <= 1
        for result in results
    )

