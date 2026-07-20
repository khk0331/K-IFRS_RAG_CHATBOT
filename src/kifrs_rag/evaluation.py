import json
from dataclasses import dataclass
from pathlib import Path

from .service import RagService


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    answerable: bool
    expected_standard_id: str | None = None
    expected_paragraph_id: str | None = None


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid eval case at line {line_number}") from error
    if not cases:
        raise ValueError("evaluation set is empty")
    return cases


def evaluate(service: RagService, cases: list[EvalCase], include_details: bool = False) -> dict:
    retrieval_hits = 0
    reciprocal_rank_sum = 0.0
    answerability_hits = 0
    details = []

    for case in cases:
        results = service.retriever.search(case.question, service.top_k)
        expected = (case.expected_standard_id, case.expected_paragraph_id)
        rank = next(
            (
                index
                for index, result in enumerate(results, 1)
                if (result.chunk.standard_id, result.chunk.paragraph_id) == expected
            ),
            None,
        )
        if case.answerable and rank is not None:
            retrieval_hits += 1
            reciprocal_rank_sum += 1 / rank

        response = service.query(case.question)
        predicted_answerable = response["status"] == "answered"
        answerability_hits += predicted_answerable == case.answerable
        if include_details:
            details.append(
                {
                    "question": case.question,
                    "expected": expected,
                    "predicted_answerable": predicted_answerable,
                    "results": [
                        {
                            "standard_id": result.chunk.standard_id,
                            "paragraph_id": result.chunk.paragraph_id,
                            "score": round(result.score, 4),
                        }
                        for result in results
                    ],
                }
            )

    answerable_count = sum(case.answerable for case in cases)
    metrics = {
        "cases": len(cases),
        f"recall_at_{service.top_k}": round(
            retrieval_hits / answerable_count if answerable_count else 0.0, 4
        ),
        "mrr": round(reciprocal_rank_sum / answerable_count if answerable_count else 0.0, 4),
        "answerability_accuracy": round(answerability_hits / len(cases), 4),
    }
    if include_details:
        metrics["details"] = details
    return metrics
