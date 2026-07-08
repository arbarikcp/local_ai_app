# Module 5 — Serving Local Models

> Phase: Foundation · Bible reference: [curriculum.md §15](../../curriculum.md#15-module-5--serving-local-models)

## Goal

Understand runtime choices and serving patterns deeply enough to know *why* a model that
works from the CLI can fail from a server, and *which* serving feature (streaming,
cancellation, structured output, usage reporting) each runtime actually supports before you
depend on it.

> **Machine note:** this repo is built on a Mac that must never run a model runtime
> ([[project-local-ai-app-curriculum]] constraint). Every piece of network-calling code in
> this module is separated from its pure parsing/orchestration logic specifically so the
> parsing logic — which is where real bugs live — is fully unit-tested here, while the
> network calls themselves are honest-skip and pending the resourced 32GB Mac (confirmed as
> the target execution machine). See `reports/module_05_runtime_serving_matrix.md`.

## 1. Direct CLI use

The simplest possible serving pattern: `ollama run <model>` or a raw `llama.cpp` binary
invocation. Good for interactive exploration, bad for anything an application depends on —
there's no stable API contract, and behavior can differ from the same model served over HTTP
(see the CLI-vs-server gotcha below).

## 2. Local HTTP API

Every runtime this course uses exposes (or can expose) a local HTTP API:

| Runtime | Native API | Port (default) |
|---|---|---|
| Ollama | `POST /api/generate`, `/api/chat`, `/api/show`, `/api/tags` | 11434 |
| llama.cpp (`llama-server`) | OpenAI-compatible `/v1/chat/completions`, plus native `/completion` | 8080 |
| llama-cpp-python `[server]` | OpenAI-compatible `/v1/chat/completions` | 8000 (configurable) |
| MLX / mlx-lm | No built-in server — library calls in-process, or wrap with your own FastAPI app | n/a |

## 3. Streaming responses

Both Ollama's native API and OpenAI-compatible servers can stream tokens as they're
generated instead of waiting for the full completion. This matters for two different reasons
that are easy to conflate:

- **UX**: streaming makes a chat UI feel responsive — the user sees tokens arrive instead of
  staring at a spinner.
- **Measurement**: streaming is the only way to measure *true* TTFT (time to the first
  generated token). Module 1's non-streaming TTFT was necessarily an approximation (load
  duration + prompt-eval duration, since the non-streaming endpoint doesn't expose a
  first-token timestamp). This module's `ollama_streaming.py` measures the real thing —
  wall-clock time to the first streamed chunk.

Ollama's `/api/generate` streams newline-delimited JSON (NDJSON): one JSON object per line,
each containing an incremental `response` text fragment, until a final object with
`"done": true` carrying the same usage fields as the non-streaming response.

## 4. OpenAI-compatible APIs

llama.cpp's `llama-server` and `llama-cpp-python[server]` both expose an
OpenAI-compatible `/v1/chat/completions` endpoint, which means the standard `openai` Python
client works against a fully local server by pointing `base_url` at it (Module 2 already
built this smoke test). This is valuable specifically because it means application code
written against the OpenAI client shape can swap between a local server and a hosted API by
changing only `base_url` and `api_key` — useful for the local inference gateway (Project 5).

## 5. Runtime lifecycle

A model isn't just "running" or "not running" — there's a lifecycle:

```text
not loaded -> loading (reads weights into memory) -> resident (ready to serve)
           -> serving (actively generating) -> idle -> unloaded (evicted)
```

Ollama manages this automatically via a `keep_alive` setting (default 5 minutes of
idleness before unload); llama.cpp-style servers typically keep the model resident for the
server's whole lifetime once loaded, with no automatic eviction. This difference matters for
both latency (a request after Ollama has unloaded the model pays a full reload) and memory
(a llama.cpp server holds its model resident even when idle, occupying that RAM the whole
time it runs).

## 6. Model warmup

The **first** request to a freshly-loaded model is measurably slower than subsequent
requests — not just because the model has to load, but because of allocator warm-up,
compute-graph construction/caching, and (on Apple Silicon) Metal shader compilation on first
use. `warmup_experiment.py` in this module measures cold-vs-warm TTFT directly rather than
asserting this from folklore.

## 7. Model unloading

Ollama's `keep_alive` parameter (per-request or server-wide) controls how long a model stays
resident after its last use — set to `0` to unload immediately after a response, or a longer
duration to keep it warm for a session of requests. Get this wrong in either direction: too
short and every request pays reload cost; too long (or `-1` for "never") and an idle model
keeps occupying RAM another module's request might need (Module 6.5's concurrency/serving
budget concern, previewed here).

## 8. Prompt caching

Distinct from the response/semantic caching Module 6.5 covers at the application layer: this
is runtime-level **prompt-prefix caching** — reusing the KV-cache computation for a prompt
prefix shared across requests (a stable system prompt, for instance) so the runtime doesn't
recompute attention for tokens it has already processed. Runtime support and cache-hit
behavior differ: llama.cpp-family runtimes generally support this via slot/prefix reuse;
Ollama's support and controls have evolved across versions — treat as "verify per version,"
not a guaranteed feature, and confirm in the deliverable matrix once tested on the resourced
Mac.

## 9. Request cancellation

If a client disconnects mid-stream (closes the HTTP connection), a well-behaved server
should stop generating rather than waste compute on a response nobody will read. This
module's `ollama_streaming.py` includes a cancellation demo that closes the stream early and
measures elapsed time at cancellation — with an explicit caveat (see Gotchas) that
client-side elapsed time is a proxy, not direct proof the server actually freed compute.

## 10. Error handling

Different runtimes signal errors differently: HTTP status codes, JSON error bodies with
different shapes, connection resets, or (worst case) a hung connection that only a client
timeout catches. Module 1's `ollama_probe.py` already wraps every httpx-level failure into a
single `OllamaUnavailable` — deliberately coarse for now, since curriculum.md §16's full
error taxonomy (`RequestTimeout`, `ModelNotLoaded`, `ModelOutOfMemory`, etc., each
distinguishable) is Module 6's job, not this one's.

## 11. Runtime-specific behavior

The point of this whole module: **do not assume feature parity across runtimes.** Structured
output support, grammar support, token-counting endpoints, streaming format, cancellation
behavior, and usage-reporting fields all differ. `feature_matrix.py` turns this from received
wisdom into a documented, versioned table — see the Deliverable section.

## Runtime comparison

| Runtime | Strength | Weakness |
|---|---|---|
| Ollama | ergonomic, quick model management, local API | less transparent than llama.cpp for low-level tuning |
| llama.cpp | excellent low-level control, GGUF ecosystem | more setup and tuning required |
| llama-cpp-python | Python-friendly, OpenAI-compatible server | build/config issues on Mac can happen |
| MLX / mlx-lm | Apple Silicon-native, good for advanced Mac path | narrower ecosystem than GGUF/Ollama |
| Transformers | standard ML ecosystem | often heavier for local app runtime |

## Serving patterns

**Pattern 1 — Direct app calls runtime** (`App -> Ollama/llama.cpp`): good for demos and
local tools. This is what every lab script in Modules 1-4 has done via `ollama_probe.py`.

**Pattern 2 — Local AI gateway** (`App -> AI Gateway -> runtime`): best for production-like
applications — this is where Module 6's `LLMRuntime` abstraction and Project 5's inference
gateway live.

**Pattern 3 — Multiple runtimes behind a router** (`App -> Gateway -> model router ->
Ollama / llama.cpp / MLX`): best for teaching model routing, fallback, and experiments —
Module 6.5 and Project 5 territory.

This module only *studies* the runtimes Pattern 1 talks to directly; it does not build
Patterns 2 or 3.

## Hands-on labs

All labs need a live runtime for their network-calling half; every lab's parsing/
orchestration logic is unit-tested here regardless, and network calls honest-skip on this
machine.

1. **Start each runtime through a repeatable command** — `scripts/module_05/serve_ollama.sh`,
   `scripts/module_05/serve_llamacpp.sh` (reviewed, not executed here, same pattern as
   Module 2's `setup_mac.sh`).
2. **Call each runtime through its native API** — `ollama_streaming.py` (native
   `/api/generate` streaming), `ollama_metadata.py` (native `/api/show`).
3. **Call OpenAI-compatible servers through the OpenAI Python client** —
   `llamacpp_openai_streaming.py`, extending Module 2's non-streaming smoke test to streaming.
4. **Add streaming response handling for prose output** — both 2 and 3 above.
5. **Add timeout, cancellation, and warmup experiments** — `ollama_streaming.py`'s
   cancellation demo, `warmup_experiment.py`.
6. **Add model metadata probes** — `ollama_metadata.py`.
7. **Document which runtime features differ** — `feature_matrix.py` +
   `reports/module_05_runtime_serving_matrix.md`.

## Gotchas

- This module studies serving behavior. The reusable `LLMRuntime` interface is defined once
  in Module 6 and reused everywhere else — nothing here should be imported by a later
  module's canonical runtime code; it's lab-local by design (same rule as Modules 1-5 so far).
- Streaming is useful for chat/prose, but structured extraction and tool arguments should
  usually be buffered and validated before use (Module 8's streaming-vs-structured rule,
  previewed here because this is where streaming is first implemented).
- A model that works from CLI may fail from a server because server context, parallelism,
  cache settings, and model residency differ — this is exactly why Lab 1's "start through a
  repeatable command" matters: an ad hoc `ollama run` session is not evidence the server
  path works.
- Native runtime APIs expose different metadata. Do not assume usage counts, stop reasons,
  or errors are normalized across runtimes — `feature_matrix.py` exists to make the actual
  differences explicit instead of assumed.
- **Cancellation verification caveat**: this module's cancellation demo measures
  client-side elapsed time after closing a stream early. That's a proxy for "the server
  stopped working," not direct proof — confirming actual server-side compute cancellation
  would require server-side instrumentation this course doesn't have access to. Documented
  as a real limitation, not silently asserted.

## Deliverable

```text
scripts/module_05/
  serve_ollama.sh
  serve_llamacpp.sh
  run_mlx_generate.py
  ollama_streaming.py
  ollama_metadata.py
  warmup_experiment.py
  llamacpp_openai_streaming.py
  feature_matrix.py
reports/module_05_runtime_serving_matrix.md
```

Curriculum's literal deliverable paths are `reports/runtime_serving_matrix.md`,
`scripts/serve_ollama.sh`, `scripts/serve_llamacpp.sh`, `scripts/run_mlx_generate.py` at the
repo root; this build places module-specific scripts under `scripts/module_05/` per the
repo-wide convention from Modules 1-4.

The matrix must document feature support and observed behavior for each runtime. On this
machine, "observed behavior" is honestly split into "verified from public documentation"
(possible now) vs. "measured directly" (pending the resourced 32GB Mac) — see the report.
