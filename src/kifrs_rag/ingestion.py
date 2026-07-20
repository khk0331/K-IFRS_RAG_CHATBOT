import json
from pathlib import Path

from .models import Chunk


REQUIRED_FIELDS = {
    "standard_id", "paragraph_id", "title", "effective_date", "source", "text"
}


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load already structured paragraphs and fail closed on incomplete metadata."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("document root must be a list")

    chunks: list[Chunk] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"item {index} missing fields: {sorted(missing)}")
        key = (
            str(item["standard_id"]),
            str(item["paragraph_id"]),
            int(item.get("chunk_index", 0)),
        )
        if key in seen:
            raise ValueError(f"duplicate paragraph chunk: {key[0]} {key[1]}#{key[2]}")
        seen.add(key)
        text = " ".join(str(item["text"]).split())
        if not text:
            raise ValueError(f"empty paragraph: {key[0]} {key[1]}")
        chunks.append(Chunk(**{**item, "paragraph_id": key[1], "text": text}))
    return chunks
