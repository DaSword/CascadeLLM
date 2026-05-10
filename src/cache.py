from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np

from config import SETTINGS


@dataclass
class CacheHit:
    query: str
    answer: str
    similarity: float
    age_seconds: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id        INTEGER PRIMARY KEY,
    query     TEXT NOT NULL,
    answer    TEXT NOT NULL,
    doc_ids   TEXT NOT NULL DEFAULT '[]',
    embedding BLOB NOT NULL,
    created   REAL NOT NULL,
    last_hit  REAL NOT NULL,
    hits      INTEGER NOT NULL DEFAULT 0,
    deleted   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entries_last_hit ON entries(last_hit);
CREATE INDEX IF NOT EXISTS idx_entries_deleted ON entries(deleted);
"""


class SemanticCache:
    def __init__(self, db_path: Path, dim: int) -> None:
        self.dim = dim
        self.threshold = SETTINGS.cache.similarity_threshold
        self.ttl_seconds = SETTINGS.cache.ttl_days * 86_400
        self.max_entries = SETTINGS.cache.max_entries
        self._lock = threading.Lock()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)

        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(max_elements=max(self.max_entries, 1024), ef_construction=200, M=16)
        self._index.set_ef(64)
        self._reload_index()

    def _reload_index(self) -> None:
        rows = self._db.execute(
            "SELECT id, embedding FROM entries WHERE deleted = 0"
        ).fetchall()
        if not rows:
            return
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        if vecs.shape[1] != self.dim:
            return
        if self._index.get_max_elements() < len(ids):
            self._index.resize_index(max(len(ids) * 2, 1024))
        self._index.add_items(vecs, ids)

    def _evict_if_needed(self) -> None:
        count = self._db.execute(
            "SELECT COUNT(*) FROM entries WHERE deleted = 0"
        ).fetchone()[0]
        if count < self.max_entries:
            return
        oldest = self._db.execute(
            "SELECT id FROM entries WHERE deleted = 0 ORDER BY last_hit ASC LIMIT 1"
        ).fetchone()
        if oldest:
            self.invalidate(oldest[0])

    def lookup(self, embedding: np.ndarray) -> CacheHit | None:
        if self._index.get_current_count() == 0:
            return None
        with self._lock:
            try:
                labels, distances = self._index.knn_query(embedding, k=1)
            except RuntimeError:
                return None
        label = int(labels[0][0])
        similarity = 1.0 - float(distances[0][0])
        if similarity < self.threshold:
            return None
        row = self._db.execute(
            "SELECT query, answer, created FROM entries WHERE id = ? AND deleted = 0",
            (label,),
        ).fetchone()
        if row is None:
            return None
        query, answer, created = row
        age = time.time() - created
        if age > self.ttl_seconds:
            self.invalidate(label)
            return None
        self._db.execute(
            "UPDATE entries SET last_hit = ?, hits = hits + 1 WHERE id = ?",
            (time.time(), label),
        )
        self._db.commit()
        return CacheHit(query=query, answer=answer, similarity=similarity, age_seconds=age)

    def put(
        self,
        query: str,
        embedding: np.ndarray,
        answer: str,
        doc_ids: list[str] | None = None,
    ) -> int:
        now = time.time()
        with self._lock:
            self._evict_if_needed()
            cur = self._db.execute(
                "INSERT INTO entries (query, answer, doc_ids, embedding, created, last_hit) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    query,
                    answer,
                    json.dumps(doc_ids or []),
                    embedding.astype(np.float32).tobytes(),
                    now,
                    now,
                ),
            )
            rid = int(cur.lastrowid)
            self._db.commit()
            try:
                self._index.add_items(embedding.reshape(1, -1), np.array([rid]))
            except RuntimeError:
                self._index.resize_index(self._index.get_max_elements() * 2)
                self._index.add_items(embedding.reshape(1, -1), np.array([rid]))
        return rid

    def invalidate(self, entry_id: int) -> None:
        self._db.execute("UPDATE entries SET deleted = 1 WHERE id = ?", (entry_id,))
        self._db.commit()
        try:
            self._index.mark_deleted(entry_id)
        except RuntimeError:
            pass

    def invalidate_by_doc(self, doc_id: str) -> int:
        rows = self._db.execute(
            "SELECT id, doc_ids FROM entries WHERE deleted = 0"
        ).fetchall()
        n = 0
        for rid, raw in rows:
            if doc_id in json.loads(raw):
                self.invalidate(rid)
                n += 1
        return n

    def existing_queries(self) -> set[str]:
        rows = self._db.execute(
            "SELECT query FROM entries WHERE deleted = 0"
        ).fetchall()
        return {r[0] for r in rows}

    def stats(self) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM entries WHERE deleted = 0"
        ).fetchone()
        return {
            "entries": int(row[0]),
            "total_hits": int(row[1]),
            "capacity": self.max_entries,
            "threshold": self.threshold,
        }
