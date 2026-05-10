from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass

from config import SETTINGS
from router import RouterResult, Tier


@dataclass
class RequestRecord:
    ts: float
    tier: str
    total_ms: float
    cost_usd: float
    edge_egress_bytes: int


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    i = max(0, min(len(xs2) - 1, int(round((len(xs2) - 1) * p))))
    return xs2[i]


class Metrics:
    def __init__(self, capacity: int = 10_000) -> None:
        self.records: deque[RequestRecord] = deque(maxlen=capacity)
        self._listeners: set[asyncio.Queue] = set()

    def record(self, result: RouterResult) -> None:
        rec = RequestRecord(
            ts=time.time(),
            tier=result.tier.value,
            total_ms=result.total_ms,
            cost_usd=result.cost_usd,
            edge_egress_bytes=result.edge_egress_bytes,
        )
        self.records.append(rec)
        payload = {"event": "request", "data": asdict(rec)}
        for q in list(self._listeners):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                self._listeners.discard(q)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    def aggregates(self) -> dict:
        if not self.records:
            return {
                "total": 0,
                "by_tier": {t.value: 0 for t in Tier},
                "cost_actual_usd": 0.0,
                "cost_baseline_usd": 0.0,
                "cost_saved_usd": 0.0,
                "edge_handled_pct": 0.0,
                "egress_bytes": 0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_by_tier": {},
            }

        by_tier = {t.value: 0 for t in Tier}
        cost_actual = 0.0
        egress = 0
        all_lat: list[float] = []
        per_tier: dict[str, list[float]] = {t.value: [] for t in Tier}
        for r in self.records:
            by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
            cost_actual += r.cost_usd
            egress += r.edge_egress_bytes
            all_lat.append(r.total_ms)
            per_tier.setdefault(r.tier, []).append(r.total_ms)

        total = len(self.records)
        baseline = total * SETTINGS.costs.cloud_gpu_usd
        edge_handled = (by_tier.get("cache", 0) + by_tier.get("edge", 0)) / total
        latency_mean = sum(all_lat) / total

        return {
            "total": total,
            "by_tier": by_tier,
            "cost_actual_usd": round(cost_actual, 6),
            "cost_baseline_usd": round(baseline, 6),
            "cost_saved_usd": round(baseline - cost_actual, 6),
            "edge_handled_pct": round(edge_handled * 100.0, 2),
            "egress_bytes": egress,
            "latency_p50_ms": round(_percentile(all_lat, 0.50), 1),
            "latency_p95_ms": round(_percentile(all_lat, 0.95), 1),
            "latency_mean_ms": round(latency_mean, 1),
            "latency_by_tier": {
                k: {
                    "p50": round(_percentile(v, 0.50), 1),
                    "p95": round(_percentile(v, 0.95), 1),
                    "mean": round(sum(v) / len(v), 1) if v else 0.0,
                    "weighted_ms": round(sum(v) / total, 1) if v else 0.0,
                    "count": len(v),
                }
                for k, v in per_tier.items()
            },
        }

    def reset(self) -> None:
        self.records.clear()
