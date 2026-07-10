# Project 5 — Local Inference Gateway

> See [PROPOSAL.md](PROPOSAL.md) for why, [ARCHITECTURE.md](ARCHITECTURE.md) for how it's built,
> [REPORT.md](REPORT.md) for measured results, [OUTRO.md](OUTRO.md) for what's next.

## Setup

No install beyond the repo's own `uv sync` (run once at the repo root). This project adds no new
dependencies — `pyyaml` is already a real, installed dependency from Module 23's config loader.

## Run the tests

```bash
uv run pytest projects/05_local_inference_gateway -q
```

## Run the gateway service

```bash
uv run uvicorn gw_api:app --app-dir projects/05_local_inference_gateway/app --port 8000
```

⚠️ With the default config (`config/app.example.yaml`), this writes to the real
`~/.local-llm-ai/gateway/` on your machine (a small SQLite request log) — same disclosed
behavior as every prior project. Set `APP_CONFIG_PATH=/path/to/config.yaml` to point at a
different data directory.

The default runtime is `FakeRuntime` (no model runtime installed on this dev machine), and the
default fallback runtime is the *same* `FakeRuntime` instance — every `/generate` call comes
back with `FakeRuntime`'s canned response until a real runtime is injected via
`build_gw_context(..., runtime=..., fallback_runtime=...)`, no other code change needed.

### Example requests

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" -d '{"task": "chat", "prompt": "hello"}'

curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" -d '{"task": "code", "prompt": "write a function"}'

curl -N -X POST http://127.0.0.1:8000/stream \
  -H "Content-Type: application/json" -d '{"task": "chat", "prompt": "hello"}'

curl -X POST http://127.0.0.1:8000/benchmark \
  -H "Content-Type: application/json" -d '{"task": "chat", "repeats": 2}'
```

Real responses from the commands above (against the default `FakeRuntime`):

```json
{"answer":"(FakeRuntime - no model runtime installed on this machine)","model_used":"llama3.1:8b-instruct","used_fallback":false,"trace_id":"d7ee7a93-d026-42bb-b09e-8016d0eb0d07"}
```

```json
{"answer":"(FakeRuntime - no model runtime installed on this machine)","model_used":"qwen2.5-coder:7b","used_fallback":false,"trace_id":"f2de489e-f01d-466a-b23e-bc6b17a48295"}
```

`model_used` proves task-based routing is real: `task: "chat"` was served by `chat`'s configured
primary model (`llama3.1:8b-instruct`), `task: "code"` by `code`'s (`qwen2.5-coder:7b`) — both
read from `config/gateway_routes.yaml`, not hardcoded per endpoint.

An unknown task is rejected before any model call:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" -d '{"task": "does-not-exist", "prompt": "hi"}'
# {"detail":"\"no route configured for task 'does-not-exist'\""}  (HTTP 404)
```

## Run the evaluation

```bash
uv run python projects/05_local_inference_gateway/evals/run_gw_eval.py
```

Unlike Projects 1/2/4's accuracy-against-a-labeled-dataset evals, a gateway has no "correct
answer" to score against — its job is correct *behavior* under real conditions. This harness
runs one real, scripted scenario per curriculum functional requirement (task routing, fallback,
streaming, timeouts, concurrency, tracing, benchmark, health) and proves the real outcome, not
just asserts it. See REPORT.md for the actual results from a real run.

## Available endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health`, `/ready` | GET | Liveness/readiness (Module 23, reused) |
| `/generate` | POST | Task-routed generation with real primary→fallback model failover |
| `/stream` | POST | Same routing/fallback, streamed chunk-by-chunk (`text/plain`) |
| `/requests/{request_id}` | GET | Retrieve a persisted gateway request log entry |
| `/benchmark` | POST | Real, freshly-measured latency/throughput for a task's configured routes |
