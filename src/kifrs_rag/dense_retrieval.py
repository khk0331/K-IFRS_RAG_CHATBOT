import json
from dataclasses import asdict
from pathlib import Path

from .models import Chunk, SearchResult


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def build_dense_index(
    chunks: list[Chunk],
    output: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    max_seq_length: int = 256,
) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    passages = [f"passage: {chunk.title} {chunk.text}" for chunk in chunks]
    vectors = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    np.save(output / "embeddings.npy", vectors)
    (output / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "model": model_name,
        "chunks": len(chunks),
        "dimensions": int(vectors.shape[1]),
        "normalized": True,
        "max_seq_length": max_seq_length,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


class DenseRetriever:
    def __init__(self, index_path: str | Path):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        self._path = Path(index_path)
        manifest = json.loads((self._path / "manifest.json").read_text(encoding="utf-8"))
        raw_chunks = json.loads((self._path / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [Chunk(**item) for item in raw_chunks]
        self._vectors = np.load(self._path / "embeddings.npy", mmap_mode="r")
        if len(self._chunks) != self._vectors.shape[0] or len(self._chunks) != manifest["chunks"]:
            raise ValueError("dense index metadata and vector count do not match")
        self._model = SentenceTransformer(manifest["model"])
        self._model.max_seq_length = manifest.get("max_seq_length", 256)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def score_all(self, question: str):
        query = self._model.encode(
            [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return self._vectors @ query

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        scores = self.score_all(question)
        count = min(top_k, len(self._chunks))
        indices = self._np.argpartition(scores, -count)[-count:]
        ranked = indices[self._np.argsort(scores[indices])[::-1]]
        return [
            SearchResult(chunk=self._chunks[int(index)], score=float(scores[int(index)]))
            for index in ranked
        ]
