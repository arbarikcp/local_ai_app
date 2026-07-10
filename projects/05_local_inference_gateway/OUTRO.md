# Outro — Project 5: Local Inference Gateway

## What this achieved

A real, running FastAPI gateway that turns curriculum's own model-routing example (a task name
→ primary/fallback model pair) into a checkable service: real task-based routing, real
primary→fallback failover proven both to work and to correctly exhaust (`NoRuntimesAvailable`
when both fail), real streaming with a deliberately narrow, well-reasoned fallback window, real
per-request timeout and concurrency enforcement, a real per-request trace with every core span
curriculum's own trace model requires, and a real benchmark command over whatever routes are
currently configured. It's also this course's clearest demonstration yet of the "compose, don't
rebuild" discipline pushed to its limit: of curriculum's 10 functional requirements, 9 had real,
already-tested infrastructure behind them before this project started (Modules 6, 6.5, 20, 21,
22, 23) — this project's genuinely new code is under 200 lines across `gw_router.py`,
`gw_model_binding.py`, and `gw_streaming.py`, wiring that already existed into the one thing
nothing in the repo did yet: deciding which model answers a request, and what happens when it
can't.

## What's still open (honest-skip, not forgotten)

- **Real model latency and quality.** Every routing, fallback, and timeout number in REPORT.md
  is real; what a real Ollama/MLX-backed primary and fallback actually cost in latency, and how
  their answers actually differ in quality, is not — `FakeRuntime` is a canned string regardless
  of model, by design. That's the load-bearing number to get once this runs on the resourced
  32GB Mac via `build_gw_context(..., runtime=..., fallback_runtime=...)` — no other code change
  needed, including for `/benchmark`, which is already generic over whatever runtime is injected.
- **No streaming fallback observability.** `stream_with_fallback()` proves the pre-vs-post-first-
  chunk distinction behaviorally, but nothing currently logs *which* branch a given stream took
  the way `/generate`'s `used_fallback` field does — a real, small gap between the two endpoints'
  observability that a future `X-Gateway-Model-Used` response header (set before the first chunk
  is written) would close.
- **A single, repo-wide `routes.yaml`.** Every deployment of this gateway shares one committed
  config; a real multi-tenant or multi-environment gateway would need per-caller or per-
  environment route overrides — not built here because curriculum's own example is a single flat
  config, and adding tenancy without a real second consumer to design against would be guessing.
- **`max_concurrent_requests` and `timeout_seconds` are both single, ungrounded defaults** (1 and
  30s respectively) — `AdmissionPolicy`'s own constructor already enforces that any value above 1
  must cite a real measurement (Module 6.5's discipline, reused unchanged here), but this project
  hasn't yet run that measurement for a *gateway* workload specifically (as opposed to the
  single-service workloads Module 6.5 measured).

## What to explore next

- **A real multi-runtime deployment, once on the resourced Mac**: configure `chat`'s primary on
  `OllamaRuntime` and its fallback on `MLXRuntime` (or vice versa) and prove a real runtime-level
  outage (not just a scripted `RuntimeUnavailable`) triggers real fallback — the direct empirical
  version of curriculum's "support multiple local runtimes" requirement, which this project
  proves structurally (DI) but hasn't yet proven against two genuinely different real backends.
- **`recommend_policy_from_measurements()` (Module 6.5, already real and tested) applied to this
  gateway specifically** — run the same concurrency-measurement lab against `/generate` once a
  real runtime is behind it, and let the gateway's own `AdmissionPolicy` carry a measured reason
  instead of the unmeasured default, closing the concurrency-limit gap named above.
- **A real LLM-as-judge or embedding-based router**, replacing the task-name-string routing key
  with a classifier that infers the task from the prompt itself (Module 13's `LocalJudge`
  pattern, or a lightweight intent classifier like Project 3's `eng_intent_classifier.py`) — a
  natural next layer once callers shouldn't be expected to already know which task bucket their
  request belongs to.
- **Response caching**, using `local_ai_core/gateway/cache.py`'s already-real
  `ResponseCache`/`SemanticCache` (confirmed available but unused by this project, since
  curriculum's own 10 requirements don't name caching) — a natural fit directly in front of
  `run_generate()`, and a genuinely interesting evaluation question: how much does a semantic
  cache actually save once real (non-fake) latency numbers exist to save against?
- **The capstone.** This gateway is explicitly the piece curriculum's own final architecture
  diagram (§39) puts in front of every other service — Projects 1-4's own `AppContext`-extending
  services are the natural next thing to sit behind it, each swapping its direct `runtime`
  injection for a call through this gateway's `/generate`/`/stream` instead.
