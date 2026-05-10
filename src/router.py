from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import embeddings
from cache import SemanticCache
from config import SETTINGS
from llm.client import (
    CLOUD_SYSTEM_PROMPT,
    EDGE_SYSTEM_PROMPT,
    GenResult,
    LLMClient,
    build_messages,
)
from llm.cloud import CloudGPU
from retriever import HybridRetriever


class Tier(str, Enum):
    CACHE = "cache"
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass
class RouterResult:
    query: str
    answer: str
    tier: Tier
    total_ms: float
    cost_usd: float
    edge_egress_bytes: int
    confidence_reason: str | None = None
    cache_similarity: float | None = None
    cache_matched_query: str | None = None


REFUSAL_MARKERS = (
    "i don't have enough information",
    "i don't know",
    "cannot answer",
    "no information",
    "not contain",
)


def _accept_edge(result: GenResult) -> tuple[bool, str]:
    text = (result.text or "").strip()
    if len(text) < SETTINGS.router.min_answer_chars:
        return False, "too_short"
    if any(m in text.lower() for m in REFUSAL_MARKERS):
        return False, "refusal"
    if (
        result.mean_logprob is not None
        and result.mean_logprob < SETTINGS.router.confidence_logprob_threshold
    ):
        return False, "low_logprob"
    return True, "ok"


def _is_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < SETTINGS.router.min_answer_chars:
        return True
    # A real refusal is short and matches a marker. A long cloud answer that
    # mentions "the docs don't contain X" while still answering is valid.
    return len(t) <= 100 and any(m in t for m in REFUSAL_MARKERS)


class Router:
    def __init__(
        self,
        cache: SemanticCache,
        retriever: HybridRetriever,
        edge_client: LLMClient,
        cloud: CloudGPU,
    ) -> None:
        self.cache = cache
        self.retriever = retriever
        self.edge = edge_client
        self.cloud = cloud

    def handle(self, query: str) -> RouterResult:
        t0 = time.perf_counter()
        q_vec = embeddings.encode_one(query)

        hit = self.cache.lookup(q_vec)
        if hit is not None:
            return RouterResult(
                query=query,
                answer=hit.answer,
                tier=Tier.CACHE,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                cost_usd=SETTINGS.costs.cache_hit_usd,
                edge_egress_bytes=0,
                cache_similarity=hit.similarity,
                cache_matched_query=hit.query,
            )

        retrievals = self.retriever.search(query, k=4)
        doc_ids = self.retriever.doc_ids_for(retrievals)
        edge_messages = build_messages(query, retrievals, EDGE_SYSTEM_PROMPT)

        edge_result = self.edge.chat(
            SETTINGS.models.edge_model,
            edge_messages,
            max_tokens=1024,
            reasoning=SETTINGS.router.edge_reasoning,
        )
        accepted, reason = _accept_edge(edge_result)

        if accepted:
            self.cache.put(query, q_vec, edge_result.text, doc_ids=doc_ids)
            return RouterResult(
                query=query,
                answer=edge_result.text,
                tier=Tier.EDGE,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                cost_usd=SETTINGS.costs.edge_slm_usd,
                edge_egress_bytes=0,
                confidence_reason=reason,
            )

        cloud_messages = build_messages(query, retrievals, CLOUD_SYSTEM_PROMPT)
        cloud_result = self.cloud.chat(
            cloud_messages,
            max_tokens=2048,
            reasoning=SETTINGS.router.cloud_reasoning,
        )

        if _is_refusal(cloud_result.text):
            return RouterResult(
                query=query,
                answer=edge_result.text,
                tier=Tier.EDGE,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                cost_usd=SETTINGS.costs.edge_slm_usd,
                edge_egress_bytes=0,
                confidence_reason="no_source_in_corpus",
            )

        self.cache.put(query, q_vec, cloud_result.text, doc_ids=doc_ids)
        egress = sum(len(m["content"].encode("utf-8")) for m in cloud_messages)
        return RouterResult(
            query=query,
            answer=cloud_result.text,
            tier=Tier.CLOUD,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            cost_usd=SETTINGS.costs.cloud_gpu_usd,
            edge_egress_bytes=egress,
            confidence_reason=reason,
        )
