from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from config import SETTINGS


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SETTINGS.models.embedding_model)


def encode(texts: list[str]) -> np.ndarray:
    with _lock:
        vecs = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


def encode_one(text: str) -> np.ndarray:
    return encode([text])[0]
