from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import embeddings
from config import DOCS_DIR


_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str


@dataclass
class Retrieval:
    chunk: Chunk
    score: float


def _rrf(scores: np.ndarray, k_const: int = 60) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return 1.0 / (k_const + ranks)


class HybridRetriever:
    def __init__(self, docs_dir: Path = DOCS_DIR) -> None:
        self.chunks: list[Chunk] = []
        self._dense: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        if docs_dir.exists():
            self._load(docs_dir)

    def _load(self, docs_dir: Path) -> None:
        for path in sorted(docs_dir.glob("*.md")):
            doc_id = path.stem
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text else doc_id
            for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
                self.chunks.append(Chunk(doc_id=doc_id, title=title, text=para))
        if not self.chunks:
            return
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])
        self._dense = embeddings.encode([c.text for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[Retrieval]:
        if not self.chunks:
            return []
        assert self._dense is not None and self._bm25 is not None

        q_dense = embeddings.encode_one(query)
        dense_scores = self._dense @ q_dense
        sparse_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))

        fused = _rrf(dense_scores) + _rrf(sparse_scores)
        top = np.argsort(-fused)[:k]
        return [Retrieval(chunk=self.chunks[i], score=float(fused[i])) for i in top]

    def doc_ids_for(self, retrievals: list[Retrieval]) -> list[str]:
        seen: list[str] = []
        for r in retrievals:
            if r.chunk.doc_id not in seen:
                seen.append(r.chunk.doc_id)
        return seen
