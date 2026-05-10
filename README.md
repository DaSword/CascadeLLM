# Cascade

A live demo of **cascading LLM inference**: serve most questions from a local cache or a small edge model, and only escalate the genuinely hard ones to a powerful cloud model. The result is **cheaper, faster, and more private** than sending every request to the cloud.

The demo simulates a 100-person company whose employees ask a chatbot questions about internal docs (HR policies, IT runbooks, expense rules, on-call schedules). Most of those questions are repetitive ("How do I submit a PTO request again?") or close paraphrases of each other. Cascade exploits that fact to handle most traffic locally and only escalate the questions that actually need a frontier model.

## How it works

Today, every chatbot question typically hits a big AI model in a faraway data center — slow, expensive, and your question leaves your network. Cascade replaces that with a **three-tier funnel** running near the user:

1. **Cache** — if someone already asked this (or something semantically similar), serve the saved answer instantly. Costs nothing.
2. **Edge** — if not cached, a small AI model running on a regular CPU answers using your internal docs. Fast and private; the question never leaves the building.
3. **Cloud** — only the questions the small model can't handle are forwarded to a large cloud AI (Google Gemini). Real cost, real network trip, but it's a small fraction of total traffic.

A live dashboard shows, in real time, how much money you saved vs. sending every question to the cloud, how fast each tier responded, and how many bytes ever left the building.

## What you'll see on the dashboard

- **$ saved vs all-cloud** — the headline number. How much cheaper this is than the naive "send everything to the cloud" baseline.
- **% handled at edge** — the share of questions that never needed the cloud. Higher is better.
- **Median response time** — how snappy the assistant feels.
- **Tier funnel** — pie chart of cache vs. edge vs. cloud requests.
- **System panel** — current edge/cloud model, cache occupancy, similarity threshold.
- **Live demo console** (right sidebar) — type a question and see exactly which tier handled it, how long it took, and what the cached match looked like (if any).
<img width="2522" height="1318" alt="image" src="https://github.com/user-attachments/assets/fa22f78e-40e4-44f7-ba7b-6d1e5d0b41f7" />

## A 5-minute live demo flow

1. **Open the dashboard.** Counters start at zero, but the **system panel** already shows ~64 cache entries — auto-seeded on boot. Briefly explain the three tiers.
2. **Type a question in the right sidebar** — e.g., *"How many vacation days do I get?"* — and hit **Ask**. It lights up the **CACHE** tier, returns in ~10 ms, and costs essentially nothing.
3. **Type a paraphrase of the same question** — e.g., *"What's our PTO policy?"*. Still hits the cache: the system understands they mean the same thing.
4. **Type something unusual** — e.g., *"Who wrote Hamlet?"*. Watch it cascade: cache (miss) → edge (refuses, not in our docs) → cloud (Gemini answers). The tier changes to **CLOUD**, cost rises, latency jumps. This is the only request that left the building.
5. **Run the replay simulator.** The cumulative-cost chart visibly diverges from the all-cloud baseline. Land on the headline: *"$X saved, Y% of questions handled locally, Z% never left the edge."*

## Why this matters

| Without Cascade | With Cascade |
|---|---|
| Every question hits the cloud | ~70% never leaves the building |
| Pay full GPU price every time | Pay only for the hard ~30% |
| Slow trip across the internet on every request | Most answers in milliseconds |
| Every question is logged centrally | Most never touch a third party |

The savings compound: the more questions a company's employees ask, the bigger the gap.

## The engine room

- **Edge LLM**: Google's [Gemma 4 E2B](https://huggingface.co/google/gemma-4-e2b) (~2B-parameter model, runs on CPU via LM Studio)
- **Cloud LLM**: Google's [Gemini 2.5 Flash-Lite](https://ai.google.dev/gemini-api/docs)
- **Embeddings**: [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) — 33M params, 384-dim. Used for cache similarity lookups and document retrieval.
- **Cache**: SQLite for durability + [hnswlib](https://github.com/nmslib/hnswlib) for fast cosine-similarity nearest-neighbor search
- **Retrieval**: hybrid BM25 + semantic search over Markdown files in `data/docs/`
- **Routing**: a confidence policy on edge logprobs and answer length decides whether to escalate to cloud

## Customizing for your own demo

- **Replace the docs.** Drop your own Markdown files into `data/docs/` (the existing ones are HR/eng samples for a fake "Akamai" company). The system will index them on next start.
- **Replace the seed questions.** Edit `data/seed_queries.json` to pre-seed the cache. Cascade auto-seeds from this file whenever it boots with an empty cache, so just delete `data/cache.sqlite` to force a re-seed.
- **Change the costs.** Edit `src/config.py` (`CostConfig`) to use real per-request pricing for your target cloud GPU and edge instance.
- **Swap the embedding model.** Change `embedding_model` and `embedding_dim` in `src/config.py`. After a model swap you need to drop `data/cache.sqlite` so the cache re-seeds in the new vector space.
