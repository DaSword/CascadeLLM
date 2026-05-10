"""Replay simulator: drives traffic against the running server.

Usage:
  python scripts/replay.py --qps 4 --duration 120
  python scripts/replay.py --total 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def load_pool() -> list[tuple[str, str]]:
    seed = json.loads((ROOT / "data" / "seed_queries.json").read_text())
    pool_extra = json.loads((ROOT / "data" / "replay_pool.json").read_text())

    # Zipf-ish weighting: a few seed queries are *very* common, the rest taper.
    weighted: list[tuple[str, float, str]] = []
    n = len(seed)
    for i, item in enumerate(seed):
        weight = 1.0 / (i + 1) ** 0.6
        weighted.append((item["q"], weight, "seed"))
    for q in pool_extra["rephrased"]:
        weighted.append((q, 0.4, "rephrased"))
    for q in pool_extra["novel_escalations"]:
        weighted.append((q, 0.15, "novel"))

    random.shuffle(weighted)
    return [(q, kind) for q, _, kind in weighted], [w for _, w, _ in weighted]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--qps", type=float, default=3.0)
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = use --total")
    p.add_argument("--total", type=int, default=400)
    args = p.parse_args()

    items, weights = load_pool()
    if not items:
        print("empty pool", file=sys.stderr)
        sys.exit(1)

    client = httpx.Client(timeout=120.0)

    delay = 1.0 / args.qps if args.qps > 0 else 0.0
    started = time.time()
    sent = 0
    counts = {"cache": 0, "edge": 0, "cloud": 0}
    cost_actual = 0.0
    cost_baseline = 0.0

    while True:
        if args.duration > 0:
            if time.time() - started >= args.duration:
                break
        else:
            if sent >= args.total:
                break

        q, _ = random.choices(items, weights=weights, k=1)[0]
        try:
            r = client.post(f"{args.server}/query", json={"query": q})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            time.sleep(delay)
            continue

        sent += 1
        counts[data["tier"]] = counts.get(data["tier"], 0) + 1
        cost_actual += data["cost_usd"]
        cost_baseline += 0.012  # configured cloud cost; matches CostConfig

        if sent % 25 == 0 or sent == 1:
            saved = cost_baseline - cost_actual
            edge_pct = 100.0 * (counts["cache"] + counts["edge"]) / sent
            print(
                f"[{sent:4d}] tier={data['tier']:5s} {data['total_ms']:6.1f}ms  "
                f"cache={counts['cache']} edge={counts['edge']} cloud={counts['cloud']}  "
                f"saved=${saved:.3f} ({edge_pct:.0f}% edge)"
            )

        time.sleep(delay)

    saved = cost_baseline - cost_actual
    print("\n=== replay summary ===")
    print(f"sent          : {sent}")
    print(f"cache hits    : {counts['cache']}  ({100*counts['cache']/sent:.1f}%)")
    print(f"edge SLM      : {counts['edge']}  ({100*counts['edge']/sent:.1f}%)")
    print(f"cloud GPU     : {counts['cloud']}  ({100*counts['cloud']/sent:.1f}%)")
    print(f"actual cost   : ${cost_actual:.3f}")
    print(f"baseline cost : ${cost_baseline:.3f}")
    print(f"saved         : ${saved:.3f}  ({100*saved/cost_baseline:.1f}%)")


if __name__ == "__main__":
    main()
