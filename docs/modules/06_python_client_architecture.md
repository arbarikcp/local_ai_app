# Module 6 — Python Client Architecture

> Phase: Foundation · Bible reference: [curriculum.md §16](../../curriculum.md#16-module-6--python-client-architecture)

## Goal

Create the **one** reusable runtime abstraction every later module builds on. Modules 1–5
deliberately smoke-tested runtimes with lab-local, throwaway code so this abstraction could
be designed from evidence — real Ollama streaming formats, real `/api/show` metadata shapes,
real error behavior — rather than guessed upfront. This module is where that evidence turns
into a stable interface. **No later module may redefine it.**

> **Machine note:** this repo is built on a Mac that must never run a model runtime
> ([[project-local-ai-app-curriculum]] constraint; target execution hardware confirmed in
> Module 5 as a separate 32GB Mac). Unlike every prior module, this one does **not** need
> live-runtime honest-skip tests for its core logic — `httpx.MockTransport` lets the HTTP
> adapters be fully and rigorously tested without a real server, and dependency injection
> does the same for the MLX adapter. See §"Testing strategy" below for exactly what that does
> and does not prove.

## 1. Runtime abstraction

One `Protocol` (structural typing, not inheritance — any object with matching methods
satisfies it) that every adapter implements and every later module's application code
depends on instead of a specific runtime:

```python
class LLMRuntime(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]: ...
    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]: ...
```

Async-first, per the curriculum's explicit instruction: a FastAPI gateway that wraps
blocking model calls without care serializes every request behind the GIL/event loop,
hiding concurrency bugs until load reveals them. Every adapter in this module is `async def`
from the start, not retrofitted later.

## 2. Request/response types

`LLMRequest` and `LLMResponse` are Pydantic models (validation, serialization, and a stable
contract for free) matching the curriculum's spec exactly — see `types.py`. `ResponseFormat`
carries the `type` (`"text" | "json_schema" | "grammar"`), an optional JSON schema, and an
optional grammar string, letting a caller ask for structured output without knowing which
runtime will serve the request.

## 3. Streaming interface

`stream()` returns `AsyncIterator[str]` — plain text fragments, matching the curriculum's
protocol exactly rather than a richer event type. This is a deliberate constraint: Module 8
will establish "stream prose, buffer structure" as the rule, so the streaming interface only
needs to carry text; usage/timing metadata belongs to the final `LLMResponse` from
`generate()`, not to every stream chunk.

## 4. Prompt templates

Out of scope for this module's interface — `LLMRequest.prompt` takes an already-assembled
prompt string. Prompt template management (versioning, few-shot examples, the invariant-first
layout Module 6.5 recommends for prefix caching) is Module 7's job. This module's adapters
must not silently reformat or reorder what's in `prompt`/`system`.

## 5. Schema validation

`response_format.schema_` (Pydantic field aliased to `"schema"` in the wire format, since
`schema` collides with `BaseModel.schema()` in Pydantic v1-style APIs) carries a JSON Schema.
Full validation-and-repair pipelines are Module 8's job; this module's contract is narrower
and non-negotiable: **if an adapter cannot honor a requested `response_format`, it raises
`FeatureNotSupported` — it never silently degrades to free-form text.** Silent degradation
would be worse than an error, because the caller would believe it got constrained output
when it didn't.

## 6. Error taxonomy

```text
LLMError
  RuntimeUnavailable      # can't reach the runtime at all (connection refused/DNS/etc.)
  ModelNotLoaded          # runtime reachable, but the requested model isn't available
  ModelOutOfMemory        # runtime signaled an OOM condition
  RequestTimeout          # the request exceeded its timeout
  InvalidModelResponse    # the runtime returned something the adapter couldn't parse
  SchemaValidationError   # response_format was requested but the model's output violates it
  ToolCallValidationError # (Module 14 will raise this from a higher layer; declared here so
                          #  the taxonomy is complete now rather than extended piecemeal later)
  SafetyPolicyViolation   # (Module 22's territory; declared here for the same reason)
  ContextTooLarge         # prompt (+history) exceeds the runtime's context window
  FeatureNotSupported     # adapter cannot honor a requested response_format/capability
```

**This resolves a real gap flagged in Modules 4 and 5's reports.** Module 1's
`ollama_probe.py` wrapped every `httpx.HTTPError` into one `OllamaUnavailable` exception,
which meant Module 4's concurrency lab could only report a generic `failure_rate`, not a true
`timeout_rate`, and Module 5's cancellation demo had the same limitation. `ollama.py` in this
module maps `httpx` exceptions precisely: `httpx.ConnectError`/`httpx.ConnectTimeout` →
`RuntimeUnavailable`, `httpx.ReadTimeout`/`httpx.PoolTimeout` → `RequestTimeout`, HTTP 404
(model not found) → `ModelNotLoaded`, HTTP 5xx → `InvalidModelResponse`. Modules 1–5's
lab-local code is intentionally left as-is (it did its job for those modules); this
taxonomy is what all *future* modules use.

## 7. Retries

`base.py`'s `with_retries()` retries only the exception types that represent **transient**
conditions — `RuntimeUnavailable` and `RequestTimeout` by default — with exponential
backoff. Everything else (a schema validation failure, an unsupported feature request)
propagates immediately.

## 8. Timeouts

Every adapter accepts an explicit `timeout` and maps a runtime timeout to `RequestTimeout`,
never lets it surface as a raw `httpx.TimeoutException` or an indefinite hang.

## 9. Metrics hooks

`base.py`'s `MetricsHook` protocol is a synchronous callback invoked after every request
(success or failure) with the request, response (or `None`), error (or `None`), and latency.
`LoggingMetricsHook` implements it with Python's `logging` module, structured as key=value
fields — this is Lab 3 (structured logging) and Lab 9 (metrics hooks) satisfied by the same
small piece of infrastructure, since a metrics hook that logs *is* structured logging.

## 10. Dependency injection

Every adapter is a plain class taking its dependencies (base URL, timeout, an `httpx.Client`
or transport, a metrics hook) as constructor arguments — nothing is a global or a
module-level singleton. This is what makes `FakeRuntime` (Lab 2) and `httpx.MockTransport`
-backed testing (§"Testing strategy") possible at all, and it is what Project 5's inference
gateway will need to route between multiple runtime instances.

## Gotchas

- `temperature=0` is not a promise of exact determinism across runtimes, hardware, or
  quantization — tests must never assert exact generated strings.
- Do not assert exact strings in tests. Assert properties: schema validity, required fields,
  allowed citations, safe tool arguments — this is why every test in this module (and every
  module going forward) checks structure/type/membership, not literal text equality on
  anything a model could plausibly phrase two ways.
- Blocking calls inside an async server can serialize requests. `MLXRuntime` is the adapter
  most at risk here — `mlx_lm` is a synchronous library — so its `generate()`/`stream()`
  run the blocking call via `asyncio.to_thread` rather than calling it directly on the event
  loop.
- Normalize errors at the adapter boundary. Do not leak runtime-specific exception types
  into application logic — every adapter's public methods only ever raise `LLMError`
  subclasses, never a raw `httpx.HTTPError` or `mlx_lm` exception.

## Testing strategy

This module reaches for two different levels of test rigor, and the report is explicit about
which is which:

1. **`FakeRuntime`** (`fake.py`): a fully in-memory implementation of `LLMRuntime`, used to
   unit-test *consumers* of the abstraction deterministically, and configurable to fail in
   specific ways (a fixed error, or "fail N times then succeed") so retry logic can be
   tested without timing games.
2. **`httpx.MockTransport`**: `OllamaRuntime` and `OpenAICompatibleRuntime` are tested by
   constructing their `httpx.Client`/`AsyncClient` with a mock transport that returns canned
   responses (or raises canned transport errors) instead of hitting a real socket. This is
   httpx's own recommended testing pattern, not a shortcut — it exercises the adapter's real
   request-building, response-parsing, streaming-iteration, and error-mapping code, just not
   the real TCP/process. **What it proves:** the adapter correctly speaks the protocol it was
   written against. **What it does not prove:** that a real, currently-running Ollama or
   llama.cpp server actually behaves the way the mock assumes — that requires the resourced
   32GB Mac, same as every other module's "real measurement" gap.
3. **`MLXRuntime`**: `mlx_lm`'s `load`/`generate`/`stream_generate` functions are injected via
   constructor (same dependency-injection principle as everything else in this module), so
   tests substitute fakes without needing MLX installed or Apple Silicon.

## Hands-on labs

1. **Implement runtime abstraction** — `base.py`, `types.py`.
2. **Implement a fake runtime for deterministic unit tests** — `fake.py`.
3. **Add structured logging** — `LoggingMetricsHook` in `base.py`.
4. **Add retries for transient errors** — `with_retries()` in `base.py`.
5. **Add no-retry for deterministic validation failures** — same function; only
   `RuntimeUnavailable`/`RequestTimeout` are retryable by default.
6. **Add token usage metadata** — `LLMResponse.prompt_tokens`/`completion_tokens`, populated
   by every adapter from real runtime usage fields (never estimated — Module 1 §5's rule).
7. **Add trace IDs** — `ensure_trace_id()` in `base.py`; every adapter propagates
   `request.trace_id` into its metrics hook calls and logs.
8. **Add adapter-specific feature negotiation** — each adapter's `_translate_response_format`
   raises `FeatureNotSupported` for anything it can't honor (informed directly by Module 5's
   `feature_matrix.py`: `mlx_lm` gets `FeatureNotSupported` for `grammar` and `json_schema`,
   `OllamaRuntime` gets it for `grammar` only).

## Deliverable

```text
packages/local_ai_core/runtimes/
  base.py
  types.py
  fake.py
  ollama.py
  openai_compatible.py
  mlx.py
  errors.py
  tests/
    test_types.py
    test_errors.py
    test_base.py
    test_fake_runtime.py
    test_ollama_runtime.py
    test_openai_compatible_runtime.py
    test_mlx_runtime.py
    test_runtime_contract.py
```

`test_runtime_contract.py` is the curriculum's explicit ask: one shared suite of assertions
(response shape, streaming yields strings, tokenize returns `list[int]`, errors are always
`LLMError` subclasses) parametrized across every adapter — `FakeRuntime`, `OllamaRuntime`
(mock-transport-backed), `OpenAICompatibleRuntime` (mock-transport-backed), and `MLXRuntime`
(fake-injected) — proving the abstraction actually abstracts, not just that each adapter
works in isolation.
