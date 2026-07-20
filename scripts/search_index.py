import argparse
import json

from kifrs_rag.dense_retrieval import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the private dense index")
    parser.add_argument("question")
    parser.add_argument("--index", default="data/index/e5-small")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    results = DenseRetriever(args.index).search(args.question, args.top_k)
    print(
        json.dumps(
            [
                {
                    "standard_id": result.chunk.standard_id,
                    "paragraph_id": result.chunk.paragraph_id,
                    "chunk_index": result.chunk.chunk_index,
                    "score": round(result.score, 4),
                    "text": result.chunk.text,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
