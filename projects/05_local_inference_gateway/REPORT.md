# Report — Project 5: Local Inference Gateway

> Measured against every commitment in [PROPOSAL.md](PROPOSAL.md)'s "How success is measured"
> table. See [ARCHITECTURE.md](ARCHITECTURE.md) for what each number is measuring.

## Status: complete

All 10 curriculum functional requirements from curriculum.md §38 are met, almost entirely
through real reuse of Modules 6, 6.5, 20, 21, 22, and 23's already-built infrastructure — this
project's own new code is task-based routing, the runtime↔model binding adapter, streaming with
pre-first-chunk fallback, and the per-request gateway log. No honest-skip surface beyond the
model runtime itself (`FakeRuntime`, this repo's standing default since Module 6) — routing,
fallback, timeout enforcement, concurrency limiting, tracing, benchmarking, and health checks all
run for real.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `config/gateway_routes.yaml` | — | Curriculum's own routes shape, real model_ids from `models/MODEL_CATALOG.md` |
| `schemas/gw_schemas.py` | 6 | API request/response shapes |
| `app/gw_router.py` | 5 | Task→route resolution, startup-time validation against the real `ModelRegistry` |
| `app/gw_model_binding.py` | 5 | `ModelBoundRuntime` composes unmodified into the real, reused `FallbackRuntime` |
| `app/gw_streaming.py` | 4 | Streaming with pre-first-chunk fallback; a post-first-chunk failure proven to propagate, not silently retry |
| `app/gw_storage.py` | 5 | Real SQLite per-request log, upsert-free append semantics |
| `app/gw_service.py` | 15 | The composition root: real routing, fallback, timeout, concurrency, and trace-completeness proofs |
| `app/gw_api.py` | 13 | Every endpoint via `TestClient`, including a real streaming read |
| `evals/run_gw_eval.py` | 2 | The full 12-scenario behavioral proof harness |

**55 new tests this project.** 2115 total across the repo, 2 correctly-skipped, all passing.
`ruff check projects/05_local_inference_gateway/` clean.

## Real proof: every curriculum requirement, one scripted scenario each

```
# Evaluation against curriculum's 10 gateway functional requirements

**12/12 scenarios passed.**

- [PASS] 1. Multiple local runtimes — two independently injected FakeRuntime instances back primary/fallback
  - served by runtime A, answer='answer from runtime A'
- [PASS] 2. Model registry — routes.yaml validated against the real 10-entry MODEL_CATALOG.md
  - registry has 10 entries
- [PASS] 3. Task-based routing — task='code' is served by code's configured primary model
  - model_used='qwen2.5-coder:7b'
- [PASS] 3b. Unknown task rejected — a task with no configured route is rejected, not silently misrouted
  - TaskNotFoundError raised before any model call
- [PASS] 4. Streaming — a real chunk-by-chunk delivery, not one buffered blob
  - received 3 chunks
- [PASS] 6. Fallback model — a failing primary falls through to the configured fallback model
  - used_fallback=True, model_used='qwen2.5:1.5b-instruct'
- [PASS] 6b. Both models failing — a failing primary AND fallback raises NoRuntimesAvailable and is logged
  - NoRuntimesAvailable raised, logged status='no_runtimes_available'
- [PASS] 5. Timeouts — a 50ms call against a 10ms timeout raises and is logged
  - RequestTimeout raised, logged status='timeout'
- [PASS] 7. Concurrency limit — max_queue_size=0 rejects a second concurrent request with QueueFullError
  - outcomes: ['ok', 'QueueFullError']
- [PASS] 8. Trace logging — every request produces a real span tree with all core steps, persisted by trace_id
  - missing_core_steps=[], span_names=['input_validation', 'model_call', 'final_response']
- [PASS] 9. Benchmark — run_gw_benchmark() returns real, freshly-measured latency for chat's primary+fallback
  - [('chat-primary', 6.33), ('chat-fallback', 5.81)]
- [PASS] 10. Health checks — run_liveness_check()/run_readiness_check() both pass against the real gateway context
  - liveness=True, readiness=True
```

Two scenarios beyond curriculum's numbered 10 (`3b`, `6b`) were added deliberately: proving an
*unknown* task is rejected (not just that a known one routes correctly) and proving *both*
models failing raises `NoRuntimesAvailable` (not just that fallback alone works) — the same
"prove the failure case too, not only the happy path" discipline Project 3's eval established.

## A real design constraint surfaced by building streaming fallback for real

`FallbackRuntime` (Module 20) has no `stream()` method — confirmed by survey before writing any
code, not discovered by trial and error. Building `stream_with_fallback()` required deciding
*when* fallback is still a legitimate operation for a stream: **only before the first chunk has
reached the caller.** A failure after streaming has already started is not the same failure mode
as a request that never got a response — a client that has already rendered "The refund amount
is 4" cannot cleanly have that silently replaced by a different model's answer starting over.
`test_gw_streaming.py::test_a_post_first_chunk_failure_propagates_instead_of_retrying` proves
this for real with a custom test double whose stream yields one real chunk before raising — not
a documented aspiration, a passing test.

## Honest-skip surface

- **Real model quality/latency.** Every routing, fallback, timeout, and concurrency number in
  this report is real; what a real Ollama/MLX-backed route's actual latency and answer quality
  look like is not — `FakeRuntime` returns the same canned string regardless of the configured
  model, by design (this repo's standing default since Module 6). Enabling it for real is a
  documented, zero-other-code-change step: `build_gw_context(..., runtime=..., fallback_runtime=...)`.
- **The `extraction`/`chat` task both routing to `category: chat` catalog entries.** Documented
  as a deliberate, ordinary choice in ARCHITECTURE.md (the real catalog has no separate
  `extraction` category), not a workaround discovered after the fact.
