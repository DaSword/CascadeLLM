from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import embeddings
from cache import SemanticCache
from config import CACHE_DB, DATA_DIR, SETTINGS, WEB_DIR
from llm.client import LLMClient
from llm.cloud import CloudGPU
from metrics import Metrics
from retriever import HybridRetriever
from router import Router


class QueryRequest(BaseModel):
    query: str


class State:
    cache: SemanticCache
    retriever: HybridRetriever
    edge: LLMClient
    cloud: CloudGPU
    router: Router
    metrics: Metrics


state = State()


def _seed_cache(cache: SemanticCache) -> int:
    path = DATA_DIR / "seed_queries.json"
    if not path.exists():
        return 0
    items = json.loads(path.read_text())
    existing = cache.existing_queries()
    items = [it for it in items if it["q"] not in existing]
    if not items:
        return 0
    vecs = embeddings.encode([it["q"] for it in items])
    for it, vec in zip(items, vecs):
        cache.put(it["q"], vec, it["a"], doc_ids=[it.get("doc", "")])
    return len(items)


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.cache = SemanticCache(CACHE_DB, dim=SETTINGS.models.embedding_dim)
    if state.cache.stats()["entries"] == 0:
        _seed_cache(state.cache)
    state.retriever = HybridRetriever()
    state.edge = LLMClient(SETTINGS.models.edge_url)
    state.cloud = CloudGPU()
    state.router = Router(state.cache, state.retriever, state.edge, state.cloud)
    state.metrics = Metrics()
    yield
    state.edge.close()
    state.cloud.close()


app = FastAPI(title="Edge LLM Demo", lifespan=lifespan)


@app.post("/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, state.router.handle, req.query)
    state.metrics.record(result)
    return {
        "query": result.query,
        "answer": result.answer,
        "tier": result.tier.value,
        "total_ms": round(result.total_ms, 1),
        "cost_usd": result.cost_usd,
        "edge_egress_bytes": result.edge_egress_bytes,
        "confidence_reason": result.confidence_reason,
        "cache_similarity": result.cache_similarity,
        "cache_matched_query": result.cache_matched_query,
    }


@app.get("/metrics")
async def metrics():
    return {
        "aggregates": state.metrics.aggregates(),
        "cache": state.cache.stats(),
        "config": {
            "cloud_is_mock": state.cloud.is_mock,
            "edge_model": SETTINGS.models.edge_model,
            "cloud_model": SETTINGS.models.cloud_model,
            "costs": {
                "cache_hit": SETTINGS.costs.cache_hit_usd,
                "edge": SETTINGS.costs.edge_slm_usd,
                "cloud": SETTINGS.costs.cloud_gpu_usd,
            },
            "threshold": SETTINGS.cache.similarity_threshold,
        },
    }


@app.get("/metrics/stream")
async def metrics_stream():
    queue = state.metrics.subscribe()

    async def gen():
        try:
            yield {"event": "snapshot", "data": json.dumps(state.metrics.aggregates())}
            while True:
                payload = await queue.get()
                yield {
                    "event": payload["event"],
                    "data": json.dumps(
                        {**payload["data"], "aggregates": state.metrics.aggregates()}
                    ),
                }
        finally:
            state.metrics.unsubscribe(queue)

    return EventSourceResponse(gen())


@app.post("/metrics/reset")
async def metrics_reset():
    state.metrics.reset()
    return {"ok": True}


@app.post("/cache/seed")
async def cache_seed():
    if not (DATA_DIR / "seed_queries.json").exists():
        raise HTTPException(404, "seed file missing")
    return {"seeded": _seed_cache(state.cache)}


@app.post("/cache/clear")
async def cache_clear():
    if CACHE_DB.exists():
        CACHE_DB.unlink()
    state.cache = SemanticCache(CACHE_DB, dim=SETTINGS.models.embedding_dim)
    _seed_cache(state.cache)
    state.router.cache = state.cache
    return {"ok": True}


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    f = WEB_DIR / "index.html"
    if not f.exists():
        return {"hint": "dashboard not built yet — POST /query to test"}
    return FileResponse(f)


def main() -> None:
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
