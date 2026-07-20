import re

from .models import SearchResult


INJECTION_PATTERNS = (
    re.compile(r"이전.{0,10}(지시|명령).{0,5}(무시|잊어)"),
    re.compile(r"ignore.{0,20}(previous|system).{0,20}(instruction|prompt)", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"(?:reveal|show|print).{0,20}(?:prompt|instruction|developer)", re.I),
    re.compile(r"(?:시스템|개발자).{0,10}(?:프롬프트|메시지).{0,10}(?:공개|보여|출력)"),
    re.compile(r"(?:act|behave)\s+as.{0,10}(?:system|developer)", re.I),
    re.compile(r"#{1,6}\s*(?:system|developer)\b", re.I),
)

SENSITIVE_PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[EMAIL]"),
    (re.compile(r"(?<!\d)01[016789]-?\d{3,4}-?\d{4}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{6}-?[1-4]\d{6}(?!\d)"), "[RRN]"),
    (re.compile(r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)"), "[CARD]"),
)


def mask_sensitive_data(text: str) -> str:
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def validate_question(question: str, max_length: int = 1000) -> str:
    clean = " ".join(question.split())
    if not clean:
        raise ValueError("질문을 입력해 주세요.")
    if len(clean) > max_length:
        raise ValueError(f"질문은 {max_length}자 이하여야 합니다.")
    if any(pattern.search(clean) for pattern in INJECTION_PATTERNS):
        raise ValueError("안전하지 않은 지시문이 감지되었습니다.")
    return mask_sensitive_data(clean)


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
