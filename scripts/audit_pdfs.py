import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


def audit(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        reader = PdfReader(path)
        pages = len(reader.pages)
        encrypted = reader.is_encrypted
        samples = sorted({0, max(0, pages // 2), max(0, pages - 1)})
        text_lengths = [len((reader.pages[index].extract_text() or "").strip()) for index in samples]
        return {
            "file": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "pages": pages,
            "encrypted": encrypted,
            "sample_text_lengths": text_lengths,
            "text_extractable": any(length > 100 for length in text_lengths),
            "error": None,
        }
    except Exception as error:  # report every damaged/unsupported file in one run
        return {
            "file": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "pages": None,
            "encrypted": None,
            "sample_text_lengths": [],
            "text_extractable": False,
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PDF integrity and text extraction")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = [audit(path) for path in sorted(args.input.glob("*.pdf"))]
    hashes: dict[str, list[str]] = {}
    for report in reports:
        hashes.setdefault(report["sha256"], []).append(report["file"])
    summary = {
        "files": len(reports),
        "pages": sum(report["pages"] or 0 for report in reports),
        "errors": sum(report["error"] is not None for report in reports),
        "encrypted": sum(report["encrypted"] is True for report in reports),
        "text_extractable": sum(report["text_extractable"] for report in reports),
        "duplicate_groups": [names for names in hashes.values() if len(names) > 1],
    }
    result = {"summary": summary, "documents": reports}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
