import math
from collections import Counter

from .dense_retrieval import DenseRetriever
from .models import Chunk, SearchResult
from .retrieval import tokenize


QUERY_EXPANSIONS = {
    "최초인식": ("취득원가", "원가", "최초 측정"),
    "최초 인식": ("취득원가", "원가", "최초 측정"),
    "후속측정": ("순실현가능가치", "장부금액", "감액", "재평가"),
    "후속 측정": ("순실현가능가치", "장부금액", "감액", "재평가"),
    "기말": ("순실현가능가치", "보고기간말", "장부금액"),
    "손상": ("회수가능액", "장부금액", "손상차손", "감액"),
}

STANDARD_HINTS = {
    "현금흐름창출단위": "K-IFRS 1036",
    "재고자산": "K-IFRS 1002",
    "현금흐름": "K-IFRS 1007",
    "법인세": "K-IFRS 1012",
    "유형자산": "K-IFRS 1016",
    "무형자산": "K-IFRS 1038",
    "충당부채": "K-IFRS 1037",
    "공정가치": "K-IFRS 1113",
    "수익": "K-IFRS 1115",
    "리스": "K-IFRS 1116",
}

SPECIAL_SCOPE_TERMS = {
    "생물자산",
    "농림어업",
    "보험계약",
    "광물자원",
    "정부보조금",
    "초인플레이션",
}


def expand_query(question: str) -> str:
    additions = [term for key, terms in QUERY_EXPANSIONS.items() if key in question for term in terms]
    return " ".join((question, *additions))


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [Counter(tokenize(f"{chunk.title} {chunk.text}")) for chunk in chunks]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(document.keys())
        count = len(chunks)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, question: str) -> list[float]:
        query = Counter(tokenize(expand_query(question)))
        scores: list[float] = []
        for document, length in zip(self.documents, self.lengths, strict=True):
            score = 0.0
            normalization = self.k1 * (
                1 - self.b + self.b * length / max(self.average_length, 1)
            )
            for token, query_frequency in query.items():
                frequency = document.get(token, 0)
                if not frequency:
                    continue
                score += (
                    self.idf.get(token, 0.0)
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + normalization)
                    * min(query_frequency, 2)
                )
            scores.append(score)
        return scores


class HybridRetriever:
    """Dense and BM25 candidate fusion followed by a lightweight domain reranker."""

    def __init__(self, index_path: str, candidate_k: int = 50):
        self.dense = DenseRetriever(index_path)
        self.chunks = self.dense.chunks
        self.sparse = BM25Retriever(self.chunks)
        self.candidate_k = candidate_k

    @staticmethod
    def _ranks(scores, count: int) -> dict[int, int]:
        import numpy as np

        count = min(count, len(scores))
        indices = np.argpartition(scores, -count)[-count:]
        ranked = indices[np.argsort(scores[indices])[::-1]]
        return {int(index): rank for rank, index in enumerate(ranked, 1)}

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        import numpy as np

        dense_scores = self.dense.score_all(question)
        sparse_scores = np.asarray(self.sparse.scores(question), dtype="float32")
        dense_ranks = self._ranks(dense_scores, self.candidate_k)
        sparse_ranks = self._ranks(sparse_scores, self.candidate_k)
        candidates = dense_ranks.keys() | sparse_ranks.keys()
        max_sparse = max((sparse_scores[index] for index in candidates), default=1.0) or 1.0
        matched_hints = [
            (term, standard) for term, standard in STANDARD_HINTS.items() if term in question
        ]
        longest_hint = max((len(term) for term, _ in matched_hints), default=0)
        hinted_standards = {
            standard for term, standard in matched_hints if len(term) == longest_hint
        }

        ranked: list[tuple[float, float, int]] = []
        for index in candidates:
            chunk = self.chunks[index]
            if hinted_standards and chunk.standard_id not in hinted_standards:
                continue
            if (
                chunk.paragraph_id.startswith(("BC", "IE", "IG"))
                and "결론도출근거" not in question
            ):
                continue
            rrf = 0.0
            if index in dense_ranks:
                rrf += 1 / (60 + dense_ranks[index])
            if index in sparse_ranks:
                rrf += 1 / (60 + sparse_ranks[index])
            lexical = float(sparse_scores[index] / max_sparse)
            standard_bonus = 0.018 if chunk.standard_id in hinted_standards else 0.0
            scope_penalty = 0.0
            for term in SPECIAL_SCOPE_TERMS:
                if term in chunk.text and term not in question:
                    scope_penalty += 0.025
            intent_bonus = 0.0
            if "측정" in question and "로 측정한다" in chunk.text:
                intent_bonus += 0.075
            if "어떻게" in question and "측정" not in chunk.text:
                scope_penalty += 0.025
            if chunk.text.endswith("측정 최초 측정"):
                scope_penalty += 0.035
            if "재고자산" in question:
                if all(term in chunk.text for term in ("취득원가", "순실현가능가치", "측정")):
                    intent_bonus += 0.055
                if "낮은 금액으로 측정" in chunk.text:
                    intent_bonus += 0.055
                if "최초" in question and "취득원가는" in chunk.text and "포함" in chunk.text:
                    intent_bonus += 0.035
            rerank_score = (
                rrf + 0.035 * lexical + standard_bonus + intent_bonus - scope_penalty
            )
            confidence = min(1.0, 0.65 * float(dense_scores[index]) + 0.35 * lexical)
            ranked.append((rerank_score, confidence, index))

        ranked.sort(reverse=True)
        selected = ranked[: min(top_k, len(ranked))]
        return [
            SearchResult(chunk=self.chunks[index], score=max(0.0, confidence))
            for _, confidence, index in selected
        ]
