# Proposal — Project 5: Local Inference Gateway

> Bible reference: [curriculum.md §38](../../curriculum.md#38-project-5--local-inference-gateway) · Structure convention: [projects/PROJECT_TEMPLATE.md](../PROJECT_TEMPLATE.md)

## Why

Every project so far (1-4) built its own composition root wrapping Module 23's `AppContext`,
each with exactly one injected `runtime: LLMRuntime` and one hardcoded model per call site.
That's the right shape for a single-purpose service, but curriculum's own final architecture
diagram (§39, the capstone) puts a "Local AI Gateway" in front of every other service, doing
model routing, fallback, and admission control once instead of each service reimplementing it.
Project 5 is that gateway: the piece every other project's `AppContext` would sit behind in a
real deployment, built as its own real, tested service rather than left as an architecture
diagram. Curriculum's own model-routing example (task → primary/fallback model pair) is the
concrete thing this project makes real and checkable.

## How

**Reused, not rebuilt** (confirmed by survey — nine of curriculum's ten functional requirements
have real, already-tested infrastructure behind them):

- `local_ai_core/runtimes/{base,fake,mlx,ollama,openai_compatible}.py` — the `LLMRuntime`
  Protocol and all four adapters (requirement 1, "multiple local runtimes"). The gateway is
  generic over whichever runtime instance(s) are injected; proven by testing against different
  runtime combinations, not by building a runtime-selection UI.
- `local_ai_core/deployment/model_registry.py`'s `ModelRegistry`/`load_model_registry()`
  (requirement 2). Every model_id a route names is validated against the real, committed
  `models/MODEL_CATALOG.md` at startup — an unknown model_id in a route is a startup error, not
  a silent runtime failure discovered on first request.
- `local_ai_core/optimization/fallback.py`'s `FallbackRuntime`/`FallbackResult` (requirement 6,
  "fallback model"), reused unchanged — only retries on `RuntimeUnavailable`/`RequestTimeout`
  (Module 6's own retryable-error taxonomy), never on a deterministic validation failure.
- `local_ai_core/runtimes/base.py`'s `LLMRuntime.stream()`, implemented for real by three of the
  four adapters (requirement 4, "streaming").
- `local_ai_core/security/tool_call_timeout.py`'s `with_timeout()` (requirement 5, "timeouts") —
  confirmed generic over any zero-arg async callable despite living in `security/`; reused for
  model calls the same way Module 22 used it for tool calls.
- `local_ai_core/gateway/admission_control.py`'s `AdmissionController`/`AdmissionPolicy`
  (requirement 7, "concurrency limit") — already a field on Module 23's `AppContext`, reused via
  `ctx.base.admission_controller`, not rebuilt.
- `local_ai_core/tracing/trace.py`'s `TraceBuilder`/`TraceSpan` (requirement 8, "trace logging"),
  reused unchanged — every prior project's `*_api.py` already threads a `trace_id` through;
  this project is the first to build real per-request span trees with it.
- `local_ai_core/optimization/benchmark_harness.py`'s `run_benchmark()`/`BenchmarkConfig`
  (requirement 9, "benchmark endpoint or command") — generic over any `LLMRuntime`, reused
  directly against the gateway's own configured routes.
- `local_ai_core/deployment/health.py`'s `run_liveness_check()`/`run_readiness_check()`
  (requirement 10, "health checks"), reused unchanged, same as every prior project.
- `local_ai_core/deployment/app_context.py`'s `AppContext`/`build_app_context()` — the
  composition root every project extends, extended the same way here.

**Built fresh** (confirmed, by survey, that nothing in the repo already does this):

- `app/gw_router.py` — `TaskRoute`, `load_routes()`: curriculum's own `routes.yaml` shape (task →
  primary/fallback model_id + max_context_tokens), validated against the real `ModelRegistry` at
  load time. Nothing in the repo maps a task string to a model pair today.
- `app/gw_model_binding.py` — `ModelBoundRuntime`: binds a `LLMRuntime` instance to one fixed
  `model_id`, so `FallbackRuntime`'s existing `list[LLMRuntime]` chain can hold "primary model on
  this runtime" and "fallback model on this runtime" as its two list entries without changing
  `FallbackRuntime` itself.
- `app/gw_streaming.py` — `stream_with_fallback()`: `FallbackRuntime` has no `stream()` method
  (confirmed by survey) and no FastAPI streaming endpoint exists anywhere in the repo to model
  from — both genuinely new here. Fallback only applies pre-first-chunk, a real, documented
  constraint (once tokens have already reached a client, silently restarting mid-stream on a
  different model is not the same operation).
- `app/gw_storage.py` — persistent per-request gateway log (task, route taken, model used,
  primary-vs-fallback, latency, trace). Nothing in the repo persists gateway-level routing
  decisions today.
- `app/gw_service.py`, `app/gw_api.py` — the composition root and FastAPI surface tying every
  reused piece above together into one real service.
- `config/gateway_routes.yaml` — a real, committed routes config using real model_ids from
  `models/MODEL_CATALOG.md`.

## What this achieves

A running FastAPI gateway (`app/gw_api.py`) that:

1. Accepts a task name (`extraction`/`code`/`chat`) and a prompt, not a raw model_id — the caller
   asks for a job, the gateway decides the model (curriculum's own task-based routing).
2. Tries the task's primary model first; on a retryable runtime failure, falls back to the task's
   configured fallback model, and records which one actually answered.
3. Supports both a buffered `/generate` and a streaming `/stream` endpoint over the same routing.
4. Enforces a per-request timeout and a configurable concurrency limit, rejecting overflow with a
   real 429/504 rather than degrading silently.
5. Produces a real per-request trace (span tree: routing decision, model call, fallback if any)
   and persists a queryable request log.
6. Exposes `/benchmark` to measure the real, currently-configured routes' latency/throughput.
7. Exposes `/health`/`/ready`, reused unchanged from Module 23.

## How success is measured

| Metric | How Project 5 measures it | Honest-skip status |
|---|---|---|
| Multi-runtime support | The same route exercised against `FakeRuntime` in different configurations (fast/slow/failing) via dependency injection | Real (structural — DI is the mechanism curriculum's requirement 1 is actually testing) |
| Task routing correctness | A request naming task `X` is proven to have been served by task `X`'s configured primary (or fallback) model_id, read back from the real per-request log | Real |
| Fallback correctness | A primary configured to fail (`FakeRuntime(fail_with=RuntimeUnavailable)`) is proven to fall through to the real configured fallback model, not silently error | Real |
| Streaming | Real chunk-by-chunk delivery over `/stream`, proven via a real `TestClient` streaming read, not just checking the final concatenated text | Real |
| Timeout enforcement | A request configured to exceed its deadline is proven to return `504`, timed with real wall-clock | Real |
| Concurrency limit | A request beyond `max_queue_size` is proven to return `429`, reusing Module 6.5's already-measured `AdmissionController` | Real |
| Trace completeness | Every request's persisted trace contains the real span names curriculum's own trace model requires (`input_validation`, `model_call`, `final_response` at minimum) | Real |
| Benchmark | `/benchmark` returns real, freshly-measured latency/throughput for the currently configured routes, not cached or asserted numbers | Real |
| Health checks | `/health`/`/ready` reused unchanged from Module 23 | Real |
| Real model quality/latency | What a real Ollama/MLX-backed route's actual latency and answer quality look like | Honest-skip — deferred to the resourced 32GB Mac, same standing default since Module 6 |

A metric only counts as "measured" in REPORT.md if it has a real, printed result from a real run
of `evals/run_gw_eval.py` or a real `curl`/`TestClient` call — not a claim.
