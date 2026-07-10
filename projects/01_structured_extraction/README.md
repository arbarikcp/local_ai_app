# Project 1 — Local Structured Extraction Service

> See [PROPOSAL.md](PROPOSAL.md) for why, [ARCHITECTURE.md](ARCHITECTURE.md) for how it's built,
> [REPORT.md](REPORT.md) for measured results, [OUTRO.md](OUTRO.md) for what's next.

## Setup

No install beyond the repo's own `uv sync` (run once at the repo root) — this project adds no
new dependencies.

## Run the tests

```bash
uv run pytest projects/01_structured_extraction -q
```

## Run the API service

```bash
uv run uvicorn extraction_api:app --app-dir projects/01_structured_extraction/app --port 8000
```

⚠️ With the default config (`config/app.example.yaml`), this writes to the real
`~/.local-llm-ai/extraction/` on your machine (a small SQLite database) — same disclosed
behavior as Module 23. Set `APP_CONFIG_PATH=/path/to/config.yaml` to point at a different data
directory.

The default runtime is `FakeRuntime` (no model runtime installed on this dev machine, per this
repo's standing constraint) — every `/extract` call will come back `needs_review` with
`"output is not valid JSON"` until a real runtime is injected (see ARCHITECTURE.md's "Machine
note" — swap the runtime via `build_extraction_context(..., runtime=<real LLMRuntime>)`, no
other code change needed).

### Example requests

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"schema_name": "invoice_v1", "text": "Invoice #A-1 for $10 from Acme."}'

curl http://127.0.0.1:8000/extractions/low-confidence

curl http://127.0.0.1:8000/extractions/<request_id>
```

Real response from the command above (against the default `FakeRuntime`):

```json
{
  "request_id": "c381a765-9bff-4f46-b20e-9802030a64c7",
  "status": "needs_review",
  "data": {},
  "confidence": "low",
  "validation_errors": ["output is not valid JSON"],
  "latency_ms": 1.73
}
```

## Run the evaluation

```bash
uv run python projects/01_structured_extraction/evals/run_extraction_eval.py
```

Runs two scenarios against the real committed `evals/extraction_dataset.jsonl` (10 examples,
both schemas): "perfect" (a scripted runtime that returns the ground-truth answer for every
example, proving the metrics score a flawless run correctly) and "imperfect" (two examples
deliberately return invalid JSON, proving the metrics catch a real failure). See REPORT.md for
the actual numbers from a real run.

## Available schemas

| `schema_name` | Schema class | Fields |
|---|---|---|
| `invoice_v1` | `local_ai_core.extraction.schemas.InvoiceExtraction` (Module 8, reused) | `invoice_number, vendor_name, invoice_date, currency, total_amount, confidence, evidence` |
| `support_ticket_v1` | `schemas/support_ticket_schema.py`'s `SupportTicketExtraction` (new) | `category, urgency, mentioned_product, customer_email, summary` |
