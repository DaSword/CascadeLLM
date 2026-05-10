from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
WEB_DIR = ROOT / "web"
CACHE_DB = DATA_DIR / "cache.sqlite"
HNSW_INDEX = DATA_DIR / "cache.hnsw"


_GEMINI_KEY = os.getenv("CLOUD_API_KEY") or os.getenv("GEMINI_API_KEY", "")
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-lite"


@dataclass
class ModelConfig:
    embedding_model: str = "Snowflake/snowflake-arctic-embed-s"
    embedding_dim: int = 384

    edge_model: str = os.getenv("EDGE_MODEL", "google/gemma-4-e2b")
    edge_url: str = os.getenv("EDGE_URL", "http://localhost:1234/v1")

    cloud_api_key: str = _GEMINI_KEY
    cloud_url: str = os.getenv("CLOUD_URL") or (_GEMINI_BASE if _GEMINI_KEY else "")
    cloud_model: str = os.getenv("CLOUD_MODEL") or (
        _GEMINI_DEFAULT_MODEL if _GEMINI_KEY else "google/gemma-4-e4b"
    )


@dataclass
class CacheConfig:
    max_entries: int = 100_000
    similarity_threshold: float = 0.82
    ttl_days: int = 7


@dataclass
class CostConfig:
    cache_hit_usd: float = 0.00002
    edge_slm_usd: float = 0.0003
    cloud_gpu_usd: float = 0.012


@dataclass
class RouterConfig:
    confidence_logprob_threshold: float = -1.2
    min_answer_chars: int = 20
    edge_reasoning: str = os.getenv("EDGE_REASONING", "off")
    cloud_reasoning: str = os.getenv("CLOUD_REASONING", "")


@dataclass
class Settings:
    models: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    router: RouterConfig = field(default_factory=RouterConfig)


SETTINGS = Settings()
