# Architecture — Project 5: Local Inference Gateway

> See [PROPOSAL.md](PROPOSAL.md) for why this exists and how success is measured.

## High-level

```text
Client request: {"task": "extraction", "prompt": "..."}
  -> input validation (schemas/gw_schemas.py)
  -> gw_router.resolve_route(task)                       [new — task -> TaskRoute]
       -> TaskRoute(primary_model, fallback_model, max_context_tokens)
  -> gw_model_binding.ModelBoundRuntime(runtime, primary_model)   [new]
     gw_model_binding.ModelBoundRuntime(runtime, fallback_model)  [new]
  -> FallbackRuntime([primary_bound, fallback_bound])     [reused, Module 20]
       -> with_timeout(...)                                [reused, Module 22]
       -> ctx.base.admission_controller.submit(...)         [reused, Module 6.5]
       -> TraceBuilder spans: input_validation, routing_decision, model_call, final_response
                                                             [reused, Module 21]
  -> gw_storage.save_request_log(...)                     [new]
  -> response: {"answer", "model_used", "used_fallback", "trace_id"}

Streaming: gw_streaming.stream_with_fallback()             [new — FallbackRuntime has no stream()]
  -> tries primary_bound.stream(); on a retryable error BEFORE the first chunk, retries with
     fallback_bound.stream() instead. A failure AFTER streaming has started propagates to the
     client as a stream error — restarting mid-stream on a different model is a different
     operation, not a transparent fallback (a real, documented constraint, not an oversight).

Benchmark: GET /benchmark -> run_benchmark() over the currently configured routes' primary and
  fallback models                                          [reused, Module 20]
```

**Deployment shape**: a single FastAPI process (Module 23's `AppContext` pattern extended,
same as every prior project), backed by one new SQLite database
(`~/.local-llm-ai/gateway/gateway.db`) for the per-request log. No new vector store, embedder, or
extraction schema — this project's job is routing and reliability around model calls, not a new
capability on top of them.

**Reused components, exact source**:

| Component | Source | Role here |
|---|---|---|
| `LLMRuntime`, `FakeRuntime`, `MLXRuntime`, `OllamaRuntime`, `OpenAICompatibleRuntime` | `local_ai_core/runtimes/` | the runtime(s) behind every route |
| `ModelRegistry`, `load_model_registry` | `local_ai_core/deployment/model_registry.py` | validates every route's model_ids at startup |
| `FallbackRuntime`, `FallbackResult`, `NoRuntimesAvailable` | `local_ai_core/optimization/fallback.py` | primary→fallback model chain |
| `with_timeout`, `RequestTimeout` | `local_ai_core/security/tool_call_timeout.py`, `local_ai_core/runtimes/errors.py` | per-request timeout enforcement |
| `AdmissionController`, `AdmissionPolicy`, `QueueFullError` | `local_ai_core/gateway/admission_control.py`, `queue.py` | concurrency limit |
| `TraceBuilder`, `TraceSpan`, `Trace` | `local_ai_core/tracing/trace.py` | per-request span tree |
| `run_benchmark`, `BenchmarkConfig`, `BenchmarkResult` | `local_ai_core/optimization/benchmark_harness.py` | `/benchmark` |
| `run_liveness_check`, `run_readiness_check` | `local_ai_core/deployment/health.py` | `/health`, `/ready` |
| `AppContext`, `build_app_context` | `local_ai_core/deployment/app_context.py` | composition root |
| FastAPI `get_ctx()` lazy-context pattern | Projects 1-4's own `*_api.py` files | copied exactly |

**New components** (why nothing existing covers them — see PROPOSAL.md's survey): task-based
routing, the runtime↔model binding adapter that lets `FallbackRuntime` carry two different
models, streaming with pre-first-chunk fallback, the per-request gateway log, and the routes
config itself.

## Low-level

### Routes config (`config/gateway_routes.yaml`)

Curriculum's own model-routing example shape (curriculum.md §38), populated with real model_ids
from the committed `models/MODEL_CATALOG.md`:

```yaml
routes:
  extraction:
    primary: qwen2.5:7b-instruct
    fallback: qwen2.5:1.5b-instruct
    max_context_tokens: 4096
  code:
    primary: qwen2.5-coder:7b
    fallback: qwen2.5-coder:1.5b
    max_context_tokens: 8192
  chat:
    primary: llama3.1:8b-instruct
    fallback: qwen2.5:1.5b-instruct
    max_context_tokens: 4096
```

`extraction` and `chat` both name `category: chat` catalog entries as their models — confirmed by
survey the real catalog has no separate `extraction` category; task names are routing keys, not
required to equal a catalog category, and reusing an instruction-tuned chat model for extraction
is a real, documented, ordinary choice, not a workaround.

### Task routing (`app/gw_router.py`)

```python
@dataclass(frozen=True)
class TaskRoute:
    task: str
    primary_model: str
    fallback_model: str
    max_context_tokens: int

class UnknownModelInRouteError(ValueError): ...

def load_routes(path: str | Path, *, model_registry: ModelRegistry) -> dict[str, TaskRoute]
def resolve_route(routes: dict[str, TaskRoute], task: str) -> TaskRoute  # raises KeyError-style TaskNotFoundError
```

`load_routes()` raises `UnknownModelInRouteError` immediately if a route names a `model_id` not
present in the injected `ModelRegistry` — a startup-time check, not a first-request surprise.

### Runtime↔model binding (`app/gw_model_binding.py`)

```python
@dataclass
class ModelBoundRuntime:
    runtime: LLMRuntime
    model_id: str

    async def generate(self, request: LLMRequest) -> LLMResponse: ...   # rewrites request.model
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]: ...
    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]: ...
```

Satisfies the `LLMRuntime` Protocol structurally, so `FallbackRuntime([ModelBoundRuntime(runtime,
route.primary_model), ModelBoundRuntime(fallback_runtime, route.fallback_model)])` reuses
`FallbackRuntime` completely unmodified — the only new logic is binding a fixed model_id onto
whichever runtime instance(s) are injected, via `LLMRequest.model_copy(update={"model": ...})`.

### Streaming with fallback (`app/gw_streaming.py`)

```python
async def stream_with_fallback(
    request: LLMRequest,
    *,
    primary: ModelBoundRuntime,
    fallback: ModelBoundRuntime,
    fallback_errors: tuple[type[LLMError], ...] = DEFAULT_FALLBACK_ERRORS,
) -> AsyncIterator[str]
```

Attempts `primary.stream(request)`; if the *first* `anext()` call raises a `fallback_errors`
member, switches to `fallback.stream(request)` for the entire response. Once a chunk has been
yielded from `primary`, no further fallback is attempted for that request — a later failure
propagates as a stream error to the client.

### Gateway request log (`app/gw_storage.py`)

```sql
CREATE TABLE gateway_requests (
    request_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    task TEXT NOT NULL,
    model_used TEXT NOT NULL,
    used_fallback INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL,           -- "ok" | "timeout" | "queue_full" | "no_runtimes_available"
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

Same idiom as every prior project's store: stdlib `sqlite3`, `check_same_thread=False` from the
start, frozen-dataclass records.

### API contract

| Endpoint | Method | Request | Response | Errors |
|---|---|---|---|---|
| `/generate` | POST | `{"task": str, "prompt": str}` | `{"answer", "model_used", "used_fallback", "trace_id"}` | 404 unknown task, 429 queue full, 504 timeout, 503 no runtimes available |
| `/stream` | POST | `{"task": str, "prompt": str}` | `text/event-stream` chunks | same as `/generate`, raised before the first chunk only |
| `/requests/{request_id}` | GET | — | the persisted `GatewayRequestRecord` | 404 |
| `/benchmark` | POST | `{"task": str?, "repeats": int?}` | real `BenchmarkResult` rows for the task's primary+fallback (or every configured task) | 404 unknown task |
| `/health`, `/ready` | GET | — | Module 23's existing checks, unchanged | — |

### Error handling

No new exception types beyond `UnknownModelInRouteError`/`TaskNotFoundError` (both startup/
routing-time, not model-call-time) — every model-call failure still flows through Module 6's
existing `LLMError` taxonomy:

| Internal error | HTTP status |
|---|---|
| Unknown task name | 404 |
| `QueueFullError` (Module 6.5) | 429 |
| `RequestTimeout` after `with_timeout` (Module 22, reused for model calls) | 504 |
| `NoRuntimesAvailable` (Module 20 — both primary and fallback failed) | 503 |
