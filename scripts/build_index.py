import argparse
import json
from pathlib import Path

from kifrs_rag.dense_retrieval import DEFAULT_MODEL, build_dense_index
from kifrs_rag.ingestion import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private dense retrieval index")
    parser.add_argument("--data", default="data/private/standards.json")
    parser.add_argument("--output", type=Path, default=Path("data/index/e5-small"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()
    manifest = build_dense_index(
        load_chunks(args.data),
        args.output,
        args.model,
        args.batch_size,
        args.max_seq_length,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
