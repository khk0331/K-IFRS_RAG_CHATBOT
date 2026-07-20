import argparse
import json

from kifrs_rag.evaluation import evaluate, load_cases
from kifrs_rag.ingestion import load_chunks
from kifrs_rag.retrieval import LocalRetriever
from kifrs_rag.service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and abstention")
    parser.add_argument("--data", default="data/sample/standards.json")
    parser.add_argument("--evals", default="evals/baseline.jsonl")
    parser.add_argument("--min-score", type=float, default=0.18)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retriever", choices=("local", "dense"), default="local")
    parser.add_argument("--index", default="data/index/e5-small")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    if args.retriever == "dense":
        from kifrs_rag.dense_retrieval import DenseRetriever

        retriever = DenseRetriever(args.index)
    else:
        retriever = LocalRetriever(load_chunks(args.data))
    service = RagService(retriever, min_score=args.min_score, top_k=args.top_k)
    print(
        json.dumps(
            evaluate(service, load_cases(args.evals), include_details=args.details),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
