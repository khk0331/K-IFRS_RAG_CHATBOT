import re
import subprocess
from pathlib import Path

from .models import Chunk


STANDARD_PATTERN = re.compile(r"제(?P<number>\d{4})호[_ ](?P<title>.+?)\(")
PARAGRAPH_PATTERN = re.compile(
    r"^\s*(?P<id>(?:[A-Z]{1,3}|한)?\s?\d+(?:\.\d+)*(?:[A-Z])?)\s{2,}(?P<text>\S.*)$"
)
PAGE_FOOTER_PATTERN = re.compile(r"^\s*-\s*\d+\s*-\s*$")
REPEATED_HEADER_PATTERN = re.compile(r"^\s*(?:기업회계기준서|기업회계기준해석서)\s+제\d+호\s*$")
STANDALONE_SYLLABLES = {"수", "등", "및", "그", "이", "각", "중", "전", "후"}
KOREAN_END_PATTERN = re.compile(r"([가-힣]+)$")
KOREAN_START_PATTERN = re.compile(r"^([가-힣]+)")


def _trim_trailing_heading(parts: list[str]) -> list[str]:
    if len(parts) > 1 and len(parts[-1]) <= 30 and not re.search(
        r"[.!?。:]$|[⑴-⒇]", parts[-1]
    ):
        return parts[:-1]
    return parts


def _join_wrapped_lines(parts: list[str]) -> str:
    if not parts:
        return ""
    result = parts[0]
    for part in parts[1:]:
        left_match = KOREAN_END_PATTERN.search(result)
        right_match = KOREAN_START_PATTERN.match(part)
        join_word = False
        if left_match and right_match:
            left = left_match.group(1)
            right = right_match.group(1)
            join_word = (
                (len(left) == 1 and left not in STANDALONE_SYLLABLES)
                or (len(right) == 1 and right in {"로", "을", "를", "이", "가", "은", "는", "의", "에", "도", "만", "과", "와", "다"})
            )
        result += ("" if join_word else " ") + part
    return result


def _split_long_text(text: str, max_chars: int = 1800, overlap: int = 180) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


def metadata_from_filename(path: Path) -> tuple[str, str, str]:
    match = STANDARD_PATTERN.search(path.name)
    if match:
        standard_id = f"K-IFRS {match.group('number')}"
        title = match.group("title").replace("_", " ").strip()
    elif "재무보고를_위한_개념체계" in path.name:
        standard_id = "K-IFRS CONCEPTUAL FRAMEWORK"
        title = "재무보고를 위한 개념체계"
    else:
        raise ValueError(f"기준서 번호를 파일명에서 찾을 수 없습니다: {path.name}")
    year_match = re.search(r"\((\d{4})_(?:개정|제정)", path.name)
    version = year_match.group(1) if year_match else "unknown"
    return standard_id, title, version


def parse_page_lines(lines: list[str]) -> list[tuple[str, str]]:
    """Parse numbered paragraphs from layout-preserving page text."""
    paragraphs: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() or PAGE_FOOTER_PATTERN.match(line) or REPEATED_HEADER_PATTERN.match(line):
            continue
        match = PARAGRAPH_PATTERN.match(line)
        if match and not re.search(r"\.{4,}", match.group("text")):
            if current:
                paragraphs.append(current)
            current = (match.group("id").replace(" ", ""), [match.group("text").strip()])
        elif current:
            current[1].append(line.strip())
    if current:
        paragraphs.append(current)
    return [
        (paragraph_id, _join_wrapped_lines(_trim_trailing_heading(parts)))
        for paragraph_id, parts in paragraphs
    ]


def extract_pdf_chunks(path: str | Path) -> list[Chunk]:
    path = Path(path)
    standard_id, title, version = metadata_from_filename(path)
    process = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
    )
    pages = process.stdout.decode("utf-8", errors="replace").split("\f")
    chunks: list[Chunk] = []
    seen: set[str] = set()
    current_id: str | None = None
    current_parts: list[str] = []
    current_page = 0

    def flush() -> None:
        nonlocal current_id, current_parts
        if not current_id or current_id in seen:
            current_id, current_parts = None, []
            return
        seen.add(current_id)
        cleaned_parts = _trim_trailing_heading(current_parts)
        paragraph_text = _join_wrapped_lines(cleaned_parts)
        for chunk_index, chunk_text in enumerate(_split_long_text(paragraph_text)):
            chunks.append(
                Chunk(
                    standard_id=standard_id,
                    paragraph_id=current_id,
                    title=title,
                    effective_date=version,
                    source=f"{path.name}#page={current_page}",
                    text=chunk_text,
                    chunk_index=chunk_index,
                )
            )
        current_id, current_parts = None, []

    for page_number, page_text in enumerate(pages, 1):
        for raw_line in page_text.splitlines():
            line = raw_line.rstrip()
            if (
                not line.strip()
                or PAGE_FOOTER_PATTERN.match(line)
                or REPEATED_HEADER_PATTERN.match(line)
            ):
                continue
            match = PARAGRAPH_PATTERN.match(line)
            if match and not re.search(r"\.{4,}", match.group("text")):
                flush()
                current_id = match.group("id").replace(" ", "")
                current_parts = [match.group("text").strip()]
                current_page = page_number
            elif current_id:
                current_parts.append(line.strip())
    flush()
    return chunks
