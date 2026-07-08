# Module 6 deliverable — Python client architecture report

Status: **complete.** Unlike every prior module, this one required no honest-skip labs —
`FakeRuntime` and `httpx.MockTransport` let the entire abstraction be built, exercised, and
verified without a live model runtime. What remains pending the resourced 32GB Mac is
narrower and specific: confirming a *real* Ollama/llama.cpp server actually behaves the way
the mocks assume (see "What this doesn't prove" below).

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `types.py` | 11 | `LLMRequest`/`LLMResponse`/`ResponseFormat` validation, defaults, the `schema`/`schema_` alias, serialization |
| `errors.py` | 24 | Every taxonomy member is an `LLMError` subclass, catchable specifically, preserves its cause without leaking the cause's type |
| `base.py` | 18 | Trace-ID generation/preservation, `Timer`, `NullMetricsHook`/`LoggingMetricsHook`, and `with_retries`' exponential backoff, retry-vs-no-retry selection, and exhaustion behavior |
| `fake.py` | 17 | `FakeRuntime`'s canned responses, per-model routing, token counting, `fail_with`/`fail_first_n_calls` failure injection, streaming, tokenize |
| `ollama.py` | 30 | Payload building, response parsing, **precise httpx-exception-to-taxonomy mapping** (connect vs. read vs. pool timeout), and full `generate`/`stream`/`tokenize` behavior against `httpx.MockTransport` |
| `openai_compatible.py` | 25 | Chat-message building, `response_format`/`grammar` translation, SDK-exception-to-taxonomy mapping, full adapter behavior against `httpx.MockTransport`, including the native (non-`/v1`) `/tokenize` endpoint |
| `mlx.py` | 18 | Chat-template rendering (with fallback), model-load caching, the thread+queue bridge that makes streaming genuinely incremental over a synchronous `mlx_lm` generator, error mapping |
| `test_runtime_contract.py` | 24 (22 pass, 2 skip) | **The curriculum's explicit ask**: one shared suite proving `FakeRuntime`, `OllamaRuntime`, `OpenAICompatibleRuntime`, and `MLXRuntime` are interchangeable — same response shape, streaming yields only strings, tokenize returns `list[int]` or a documented `LLMError`, and unsupported features raise `LLMError`, never a raw runtime exception |

**167 new tests this module** (165 passing + 2 correctly-skipped; 395 total across the repo,
all passing), `ruff check .` clean.

## Two real bugs this module's own tests caught

### 1. Double-retry composition risk (`openai_compatible.py`)

The `openai` SDK retries transient errors internally by default. Left alone, a caller using
`with_retries()` around `OpenAICompatibleRuntime.generate()` would compose with the SDK's own
retry loop — up to 3× (this module's default) × the SDK's own retry count in the worst case,
silently multiplying latency and obscuring real failure rates. Noticed because the adapter's
own test suite took an anomalous 3.01s to run (vs. ~0.1-0.3s for every sibling test file).
Fixed by setting `max_retries=0` on every `AsyncOpenAI` client this module constructs, with a
comment explaining why: **retry policy lives in exactly one place, `base.py`'s
`with_retries()`, applied by the caller.** Test suite runtime dropped to 0.72s after the fix.

### 2. A false "uniform capability" assumption in the contract test itself

The first version of `test_runtime_contract.py` asserted that *every* adapter rejects
`response_format.type="grammar"` with `FeatureNotSupported`. Running it immediately failed
for `OpenAICompatibleRuntime` — correctly, because real llama.cpp/llama-cpp-python servers
**do** support GBNF grammar (confirmed `yes` in Module 5's `feature_matrix.py`), so the
adapter correctly accepts it rather than rejecting it. The test's premise was wrong, not the
code. Fixed by making the "adapter rejects format X" expectation explicit per-adapter
(`UNSUPPORTED_FORMAT_BY_ADAPTER`), with `OpenAICompatibleRuntime` deliberately having no entry
and a comment explaining why — this is Module 5's entire point (runtimes differ) showing up
as a real, caught assertion rather than an assumed one.

## The failure_rate/timeout_rate gap (Modules 4 and 5) is now resolved

`ollama.py`'s `map_httpx_error` distinguishes `httpx.ConnectError`/`httpx.ConnectTimeout`
(→ `RuntimeUnavailable`) from `httpx.ReadTimeout`/`httpx.PoolTimeout` (→ `RequestTimeout`),
verified by dedicated tests for each case. Module 1's lab-local `ollama_probe.py` (used by
Modules 1-5's labs) is intentionally left as-is — it did its job for those modules — but any
*new* code should use `OllamaRuntime` from this module to get the precise taxonomy.

## What this module's testing does and does not prove

Per the theory doc's "Testing strategy" section, restated here because it matters for how
much confidence to place in this report:

- **Proven**: every adapter correctly speaks the protocol it was written against — request
  shapes, response parsing, streaming iteration, and error mapping all behave exactly as
  designed, across 165 passing tests including a shared cross-adapter contract suite.
- **Not proven**: that a real, currently-running Ollama or llama.cpp-family server actually
  behaves the way the mocks assume. `httpx.MockTransport` is httpx's own recommended testing
  pattern, not a shortcut, but it is still a mock. Confirming real-server behavior is the
  same "pending the resourced 32GB Mac" gap every other module has — narrower here because
  the adapter logic itself is already fully verified, so completing this is a matter of
  pointing `OllamaRuntime()`/`OpenAICompatibleRuntime()` at real servers and checking nothing
  about the *real* protocol differs from what the mocks assumed.

## Next steps (on the resourced Mac)

```python
# Point the adapters at real servers and re-run the notebook's demo cells for real:
from local_ai_core.runtimes.ollama import OllamaRuntime
from local_ai_core.runtimes.openai_compatible import OpenAICompatibleRuntime
from local_ai_core.runtimes.mlx import MLXRuntime

ollama_rt = OllamaRuntime()  # defaults to http://localhost:11434
response = await ollama_rt.generate(LLMRequest(model="qwen2.5:1.5b", prompt="hi"))
```

If real-server behavior ever diverges from what a mock assumed (a field renamed, an error
shape that doesn't match), that's a finding for this report, not a silent adapter bug —
update the relevant `MockTransport` handler in the tests to match reality once discovered.
