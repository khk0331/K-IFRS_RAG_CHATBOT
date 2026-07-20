import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from kifrs_rag.pdf_ingestion import extract_pdf_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse K-IFRS PDFs into private JSON chunks")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/private/standards.json"))
    parser.add_argument("--report", type=Path, default=Path("data/private/ingestion_report.json"))
    args = parser.parse_args()

    paths = sorted(args.input.glob("*.pdf"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        parsed_documents = list(executor.map(extract_pdf_chunks, paths))

    all_chunks = []
    documents = []
    for path, chunks in zip(paths, parsed_documents, strict=True):
        all_chunks.extend(chunks)
        documents.append(
            {
                "file": path.name,
                "standard_id": chunks[0].standard_id if chunks else None,
                "paragraphs": len(chunks),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(chunk) for chunk in all_chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "files": len(documents),
        "paragraphs": len(all_chunks),
        "empty_documents": [item["file"] for item in documents if item["paragraphs"] == 0],
        "documents": documents,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("files", "paragraphs", "empty_documents")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
