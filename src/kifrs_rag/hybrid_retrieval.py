import math
from collections import Counter

from .dense_retrieval import DenseRetriever
from .models import Chunk, SearchResult
from .retrieval import tokenize


STANDARD_HINTS = {
    "현금흐름창출단위": "K-IFRS 1036",
    "현금창출단위": "K-IFRS 1036",
    "사용권자산": "K-IFRS 1116",
    "리스부채": "K-IFRS 1116",
    "영업권 손상": "K-IFRS 1036",
    "재고자산": "K-IFRS 1002",
    "현금흐름": "K-IFRS 1007",
    "회계정책": "K-IFRS 1008",
    "보고기간후사건": "K-IFRS 1010",
    "보고기간후": "K-IFRS 1010",
    "조정사건": "K-IFRS 1010",
    "법인세": "K-IFRS 1012",
    "유형자산": "K-IFRS 1016",
    "종업원급여": "K-IFRS 1019",
    "차입원가": "K-IFRS 1023",
    "특수관계자": "K-IFRS 1024",
    "주당이익": "K-IFRS 1033",
    "중간재무보고": "K-IFRS 1034",
    "자산손상": "K-IFRS 1036",
    "무형자산": "K-IFRS 1038",
    "투자부동산": "K-IFRS 1040",
    "충당부채": "K-IFRS 1037",
    "사업결합": "K-IFRS 1103",
    "영업부문": "K-IFRS 1108",
    "연결재무제표": "K-IFRS 1110",
    "공동약정": "K-IFRS 1111",
    "공정가치": "K-IFRS 1113",
    "수행의무": "K-IFRS 1115",
    "지배력": "K-IFRS 1110",
    "수익": "K-IFRS 1115",
    "리스": "K-IFRS 1116",
    "보험계약": "K-IFRS 1117",
}

SPECIAL_SCOPE_TERMS = {
    "생물자산",
    "농림어업",
    "보험계약",
    "광물자원",
    "정부보조금",
    "초인플레이션",
}

ACCOUNTING_ANCHORS = {
    "회계",
    "재무",
    "자산",
    "부채",
    "자본",
    "수익",
    "비용",
    "인식",
    "측정",
    "공시",
    "손상",
    "원가",
    "가치",
    "연결",
    "리스",
}


def has_initial_intent(question: str) -> bool:
    return any(term in question for term in ("최초인식", "최초 인식", "최초 측정"))


def has_subsequent_intent(question: str) -> bool:
    return any(term in question for term in ("후속측정", "후속 측정", "기말"))


def expand_query(question: str) -> str:
    additions: list[str] = []
    initial = has_initial_intent(question)
    subsequent = has_subsequent_intent(question)
    if "재고자산" in question:
        if initial:
            additions.extend(("취득원가", "원가"))
        if subsequent:
            additions.extend(("순실현가능가치", "장부금액", "감액", "재평가"))
    if "사용권자산" in question:
        if initial:
            additions.extend(("리스개시일", "원가"))
        if subsequent:
            additions.extend(("리스개시일 후", "원가모형", "감가상각", "손상차손", "재측정"))
    if "손상" in question:
        additions.extend(("회수가능액", "장부금액", "손상차손", "감액"))
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

    def __init__(self, index_path: str, candidate_k: int = 250):
        self.dense = DenseRetriever(index_path)
        self.chunks = self.dense.chunks
        self.sparse = BM25Retriever(self.chunks)
        self.candidate_k = candidate_k

    @staticmethod
    def _ranks(scores, count: int, allowed: list[int] | None = None) -> dict[int, int]:
        import numpy as np

        pool = np.asarray(allowed if allowed is not None else range(len(scores)), dtype="int64")
        count = min(count, len(pool))
        if not count:
            return {}
        pool_scores = scores[pool]
        positions = np.argpartition(pool_scores, -count)[-count:]
        indices = pool[positions]
        ranked = indices[np.argsort(scores[indices])[::-1]]
        return {int(index): rank for rank, index in enumerate(ranked, 1)}

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        return self._search(question, top_k, use_routing=True)

    def search_broad(self, question: str, top_k: int = 12) -> list[SearchResult]:
        return self._search(question, top_k, use_routing=False)

    def _search(self, question: str, top_k: int, use_routing: bool) -> list[SearchResult]:
        import numpy as np

        initial = has_initial_intent(question)
        subsequent = has_subsequent_intent(question)
        dense_scores = self.dense.score_all(question)
        sparse_scores = np.asarray(self.sparse.scores(question), dtype="float32")
        matched_hints = [
            (term, standard) for term, standard in STANDARD_HINTS.items() if term in question
        ] if use_routing else []
        longest_hint = max((len(term) for term, _ in matched_hints), default=0)
        hinted_standards = {
            standard for term, standard in matched_hints if len(term) == longest_hint
        }
        allowed = (
            [index for index, chunk in enumerate(self.chunks) if chunk.standard_id in hinted_standards]
            if hinted_standards
            else None
        )
        dense_ranks = self._ranks(dense_scores, self.candidate_k, allowed)
        sparse_ranks = self._ranks(sparse_scores, self.candidate_k, allowed)
        candidates = dense_ranks.keys() | sparse_ranks.keys()
        max_sparse = max((sparse_scores[index] for index in candidates), default=1.0) or 1.0

        ranked: list[tuple[float, float, int]] = []
        for index in candidates:
            chunk = self.chunks[index]
            if hinted_standards and chunk.standard_id not in hinted_standards:
                continue
            if (
                chunk.paragraph_id.startswith(("BC", "IE", "IG", "IN"))
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
            if "정의" in question and "정의한다" in chunk.text:
                intent_bonus += 0.18
            if "정의" in question and "가격으로 정의한다" in chunk.text:
                intent_bonus += 0.18
            if "조정사건" in question and "수정을 요하는 보고기간후사건" in chunk.text:
                intent_bonus += 0.18
            if (
                "수행의무" in question
                and "식별" in question
                and "수행의무로 식별" in chunk.text
            ):
                intent_bonus += 0.18
            if (
                "수행의무" in question
                and "식별" in question
                and "각 약속을 하나의 수행의무로 식별" in chunk.text
            ):
                intent_bonus += 0.18
            if "지배력" in question and "다음 모두" in chunk.text and "지배" in chunk.text:
                intent_bonus += 0.18
            if "재고자산" in question:
                if all(term in chunk.text for term in ("취득원가", "순실현가능가치", "측정")):
                    intent_bonus += 0.055
                if "낮은 금액으로 측정" in chunk.text:
                    intent_bonus += 0.055
                if "최초" in question and "취득원가는" in chunk.text and "포함" in chunk.text:
                    intent_bonus += 0.035
            if "사용권자산" in question:
                if initial and all(term in chunk.text for term in ("리스개시일", "원가", "측정")):
                    intent_bonus += 0.065
                if subsequent and any(
                    term in chunk.text
                    for term in ("리스개시일 후", "원가모형", "감가상각", "재측정")
                ):
                    intent_bonus += 0.065
            rerank_score = (
                rrf + 0.035 * lexical + standard_bonus + intent_bonus - scope_penalty
            )
            confidence = 0.65 * float(dense_scores[index]) + 0.35 * lexical
            confidence += min(intent_bonus, 0.18)
            if hinted_standards:
                confidence += 0.03
            elif not any(anchor in question for anchor in ACCOUNTING_ANCHORS):
                confidence -= 0.15
            confidence = min(1.0, confidence)
            ranked.append((rerank_score, confidence, index))

        ranked.sort(reverse=True)
        selected = ranked[: min(top_k, len(ranked))]
        return [
            SearchResult(chunk=self.chunks[index], score=max(0.0, confidence))
            for _, confidence, index in selected
        ]
