from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx

from retriever import Retrieval


EDGE_SYSTEM_PROMPT = (
    "You are an internal company assistant. Answer using only the context "
    "below. If the context does not contain the answer, reply exactly: "
    "I don't have enough information to answer that.\n\n"
    "Be concise (1-3 sentences). Do not invent facts."
)

CLOUD_SYSTEM_PROMPT = (
    "Answer the user's question accurately and concisely (1-3 sentences). "
    "The optional context below contains internal company documents — use "
    "them only when they are clearly relevant to the question. Otherwise, "
    "answer from your own general knowledge. ALWAYS provide a substantive "
    "answer when one is possible from general knowledge. Do not refuse "
    "questions just because the context doesn't cover them. The only "
    "facts you must not invent are specific internal company details "
    "(policies, people, projects) that aren't in the context."
)


def build_messages(
    query: str,
    retrievals: list[Retrieval],
    system_prompt: str = EDGE_SYSTEM_PROMPT,
) -> list[dict]:
    if retrievals:
        context = "\n\n".join(
            f"[{i + 1}] {r.chunk.title}\n{r.chunk.text}"
            for i, r in enumerate(retrievals)
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user = query
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


@dataclass
class GenResult:
    text: str
    mean_logprob: float | None
    tokens: int
    duration_ms: float


class LLMClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 256,
        reasoning: str | None = None,
        logprobs: bool = True,
    ) -> GenResult:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
        if reasoning:
            payload["reasoning"] = reasoning
        t0 = time.perf_counter()
        r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} from {self.base_url}: {r.text[:500]}")
        body = r.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        choice = body["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        if not text.strip() and choice.get("finish_reason") == "length":
            text = "I ran out of tokens before producing an answer. Please try again."

        mean_lp: float | None = None
        token_lps = (choice.get("logprobs") or {}).get("content") or []
        vals = [
            t["logprob"] for t in token_lps
            if t.get("logprob") is not None and not math.isinf(t["logprob"])
        ]
        if vals:
            mean_lp = sum(vals) / len(vals)

        usage = body.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or len(token_lps) or 0)

        return GenResult(
            text=text.strip(),
            mean_logprob=mean_lp,
            tokens=tokens,
            duration_ms=elapsed_ms,
        )

    def close(self) -> None:
        self._client.close()
