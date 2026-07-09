# Module 20 — Inference Optimization Under 8–24 GB RAM

> Phase: Serving/performance foundation · Bible reference: [curriculum.md §30](../../curriculum.md#30-module-20--inference-optimization-under-824-gb-ram)

## Goal

Optimize latency, memory, and reliability for local LLM apps.

## This module is mostly a playbook, not a rebuild

Most of curriculum's 16 core topics already have real, tested implementations from earlier
modules — this module's job is to name that reuse explicitly, cite exactly which file
implements each topic, and build the genuinely new pieces the playbook needs: a **model
router**, a **fallback chain**, a **benchmark harness**, and a **performance dashboard**. Reuse
list, one line per topic already covered elsewhere:

| Topic | Already implemented in |
|---|---|
| Quantization choice | `scripts/module_04/memory_math.py` (`weights_bytes()`), `lab_4_1_quantization_comparison.py` |
| Context budgeting | `local_ai_rag/context_packers/budget_packer.py` (`ContextBudget`, `pack_context()`) for RAG context; `local_ai_core/conversation/token_budget.py` (`ConversationBudget`) for chat history |
| Streaming | `runtimes/base.py`'s `LLMRuntime.stream()` Protocol, implemented for real in `runtimes/mlx.py`/`runtimes/ollama.py` |
| Model warmup | Module 5 theory doc §5 (runtime-level `keep_alive`/residency behavior) |
| Prompt caching / response & semantic caching | `gateway/cache.py`'s `ResponseCache` |
| KV cache behavior | Module 4's `kv_cache_bytes()`, Module 6.5 theory doc §2 |
| Concurrency control | `gateway/admission_control.py`'s `AdmissionController`, `AdmissionPolicy` |
| Request queueing | `gateway/queue.py`'s `BoundedRequestQueue` |
| Timeout policies | `runtimes/errors.py`'s `RequestTimeout`, `runtimes/base.py`'s `with_retries()` |
| Reranking vs bigger model | Module 12's `CrossEncoderReranker` |

Everything in that table is cited, not re-implemented — Module 19's QLoRA precedent ("compose,
don't rebuild") applies here across ten topics at once.

## New in this module

`optimization/model_router.py`'s `route_model()` (task complexity → small/large model tier, a
real testable decision function like Module 18's `should_use_vlm()`), `optimization/fallback.py`'s
`FallbackRuntime` (an ordered chain of runtimes, real retry-to-next-on-failure),
`optimization/benchmark_harness.py`'s `run_benchmark()` (real latency/tokens-per-second
measurement over `FakeRuntime`'s `simulated_latency_ms` — genuinely computed numbers, just not
against a live model), `optimization/dashboard.py`'s `InMemoryMetricsHook` +
`PerformanceDashboard` (real p50/p95 aggregation implementing the existing `MetricsHook`
Protocol), and `optimization/prompt_compression.py`'s `compress_prompt()` (real, deterministic,
non-LLM prompt shrinking — whitespace collapse and duplicate-line removal, distinct from Module
8.5's `summarizer.py`'s LLM-based history summarization).

> **Machine note:** every new module in this table's "New in this module" section is fully
> real and testable without a model runtime — `FakeRuntime`'s `simulated_latency_ms` (Module 6.5
> precedent) is exactly what a benchmark harness needs to produce genuine timing numbers.
> Real per-runtime, per-quantization measurement against a live Ollama/MLX server stays
> honest-skip, pending the resourced 32GB Mac, same as Modules 4-6.5.

## Core topics

### 1-2. Quantization choice, context budgeting

Reused, not rebuilt — see table above.

### 3. Prompt compression

`prompt_compression.py`'s `compress_prompt()` — real whitespace collapsing (multiple blank
lines/spaces to one) and exact-duplicate consecutive line removal, reporting a real
before/after heuristic token count and reduction ratio. Distinct from Module 8.5's
`summarizer.py`, which compresses via an LLM call (lossy, needs a model); this is lossless-ish,
deterministic, and needs no model.

### 4-7. Streaming, model warmup, prompt caching, KV cache behavior

Reused, not rebuilt — see table above.

### 8-9. Concurrency control, request queueing

Reused, not rebuilt — `gateway/admission_control.py`, `gateway/queue.py`.

### 10. Timeout policies

Reused — `runtimes/errors.py`'s `RequestTimeout`, retried via `with_retries()`'s
`DEFAULT_RETRYABLE_ERRORS`.

### 11. Fallback models

`fallback.py`'s `FallbackRuntime` — wraps an ordered list of `LLMRuntime`s; `generate()` tries
each in order, catching only the same retryable error types Module 6's `with_retries()`
already recognizes (`RuntimeUnavailable`, `RequestTimeout`) and moving to the next runtime,
re-raising the last error if every runtime in the chain fails. A non-retryable error (e.g. a
validation failure) propagates immediately without wasting a fallback attempt on a
deterministic failure — same principle `with_retries()` already applies to retries.

### 12. Reranking vs bigger model

Theory, tied to Module 12's `CrossEncoderReranker` — a reranker narrows context to fewer,
better chunks so a *smaller* generation model can answer correctly; a bigger model is the more
expensive lever and should be reached for only after reranking is already in place and proven
insufficient.

### 13. Small model routers

`model_router.py`'s `route_model()` — real boolean/numeric task signals (`prompt_token_count`,
`requires_multi_step_reasoning`, `requires_tool_calls`, `output_must_be_structured`) mapped to
a `ModelTier` (`SMALL`/`LARGE`) and a specific reason, the same discipline Module 19's
`recommend_approach()` applied to the prompting/RAG/fine-tuning decision.

### 14-16. Thermal throttling, memory pressure, disk pressure

Theory only — genuinely hardware-dependent runtime behavior this dev machine can't produce
without a model actually running (same reasoning Module 4 gave for real memory sampling: the
*math* is real and testable, the *live measurement* needs a resourced Mac). Documented here as
operational guidance: thermal throttling degrades tokens/sec silently (watch `tokens_per_second`
trend, not just latency); memory pressure should trigger the "When memory is high" playbook
below before macOS starts swapping; disk pressure matters because model weight files are large
enough (multi-GB per model) that `df -h` should be part of any deployment checklist, covered
operationally in Module 23 (packaging and deployment).

## Optimization playbook

Curriculum's three playbooks (latency, quality, memory), reproduced here as the canonical
reference — each numbered step already maps to a real function or existing module, cited where
one applies:

**When latency is high:** reduce prompt tokens (`prompt_compression.py`) → reduce max output
tokens → use smaller model (`model_router.py`) → lower quantization (Module 4) → add streaming
(`runtimes/base.py`) → improve retrieval precision (Module 12) → use reranker to reduce context
(Module 12's `CrossEncoderReranker`) → cache repeated prompt prefixes (`gateway/cache.py`) →
use KV-cache/prefix reuse where supported (runtime-level) → use response/semantic caching
(`gateway/cache.py`) → use task-specific small model (`model_router.py`) → route heavy tasks to
a larger model only when needed (`model_router.py` + `fallback.py`).

**When quality is low:** improve prompt (Module 7) → add examples (Module 7) → add schema
validation (Module 8) → improve retrieval (Module 12) → add reranker (Module 12) → use better
embedding model (Module 9) → use larger model (`model_router.py`) → fine-tune only after
evaluation (Module 19's `recommend_approach()` — the same gate, restated).

**When memory is high:** reduce context (`ContextBudget`) → reduce concurrency
(`AdmissionPolicy`) → use smaller quantization (Module 4) → unload unused models (Module 5's
runtime residency behavior) → avoid multiple large runtimes running together → run embedder,
reranker, and generator sequentially unless measurement proves co-residency is safe → reduce
batch size → quantize KV cache where supported → keep `max_concurrent_requests: 1` until
benchmarks justify increasing it (`recommend_policy_from_measurements()`, Module 6.5).

## Hands-on labs

1. **Build benchmark harness** — `optimization/benchmark_harness.py`,
   `scripts/module_20/benchmark_harness_demo.py`. Real latency/tokens-per-second measurement
   over `FakeRuntime`'s simulated latency.
2. **Add context budgeter** — `scripts/module_20/context_budget_demo.py`, composing Module
   12's `ContextBudget`/`pack_context()` and Module 8.5's `ConversationBudget` for one full
   request's token accounting — no new package code, since both budgeters already exist.
3. **Add model router** — `optimization/model_router.py`,
   `scripts/module_20/model_router_demo.py`.
4. **Add fallback model** — `optimization/fallback.py`,
   `scripts/module_20/fallback_demo.py`.
5. **Add queueing** — `scripts/module_20/queueing_streaming_demo.py`, reusing Module 6.5's
   `BoundedRequestQueue` unchanged.
6. **Add streaming** — same script, reusing `FakeRuntime.stream()` unchanged.
7. **Add performance dashboard** — `optimization/dashboard.py`,
   `scripts/module_20/performance_dashboard_demo.py`.

## Deliverable

```text
packages/local_ai_core/optimization/
  model_router.py
  fallback.py
  benchmark_harness.py
  dashboard.py
  prompt_compression.py
  tests/
scripts/module_20/
  benchmark_harness_demo.py
  context_budget_demo.py
  model_router_demo.py
  fallback_demo.py
  queueing_streaming_demo.py
  performance_dashboard_demo.py
reports/module_20_inference_optimization_report.md
```
