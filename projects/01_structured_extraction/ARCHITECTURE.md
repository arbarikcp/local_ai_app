# Architecture — Project 1: Local Structured Extraction Service

> See [PROPOSAL.md](PROPOSAL.md) for why this exists and how success is measured.

## High-level

```text
Input document (JSON text field)
  -> extraction_normalization.normalize_text()        [new]
  -> extraction_prompts.build_prompt()                 [new, wraps Module 8's build_extraction_prompt]
  -> ExtractionPipeline.run()                           [reused, Module 8]
       -> local LLM (FakeRuntime on this machine)
       -> json_parsing.try_parse_json()                 [reused, Module 8]
       -> Pydantic validator (schema.model_validate)     [reused, Module 8]
       -> repair retry (<= 2 attempts)                   [reused, Module 8, max_repair_attempts=2]
       -> confidence.compute_confidence()                [reused, Module 8]
  -> extraction_storage.ExtractionStore.save()          [new]
  -> extraction_api (FastAPI)                           [new]
       -> POST /extract
       -> GET /extractions/{request_id}
       -> GET /extractions/low-confidence
```

**Deployment shape**: a single FastAPI process (Module 23's `AppContext` pattern extended, not
replaced), backed by one new SQLite database (`~/.local-llm-ai/extraction/extraction.db`),
alongside the existing `sessions`/`audit`/`adapters`/`eval_feedback` databases Module 23 already
manages. No new deployment mode — this *is* "the FastAPI local service" curriculum's own
Module 23 deployment-modes table already named as "best for backend architecture."

**Reused components, exact source**:

| Component | Source | Role here |
|---|---|---|
| `ExtractionPipeline` | `local_ai_core/extraction/pipeline.py` | prompt -> LLM -> parse -> validate -> repair -> confidence |
| `InvoiceExtraction` | `local_ai_core/extraction/schemas.py` | schema 1 (`invoice_v1`) |
| `compute_confidence`, `try_parse_json`, `chunk_text` | `local_ai_core/extraction/{confidence,json_parsing,chunking}.py` | used internally by `ExtractionPipeline`, not called directly by this project |
| `AppContext`, `build_app_context` | `local_ai_core/deployment/app_context.py` | composition root, extended with a `storage` field |
| `AppConfig`, `load_config` | `local_ai_core/deployment/config.py` | unchanged, `models.default_extraction` already exists as a config field |
| `run_startup_checks`, `run_readiness_check` | `local_ai_core/deployment/health.py` | unchanged |
| `backup_sqlite_db` | `local_ai_core/deployment/backup.py` | unchanged, pointed at the new extraction database |
| `with_retries` | `local_ai_core/runtimes/base.py` | wraps the runtime call for transport-level timeout/unavailability retry, composed around the pipeline's own semantic repair retry |
| FastAPI `get_ctx()` lazy-context pattern | `scripts/module_23/api_app.py` | copied exactly for `extraction_api.py` |

**New components** (why nothing existing covers them — see PROPOSAL.md's survey): text
normalization, persistent storage with a low-confidence query, a second schema, the FastAPI
endpoints, the evaluation command.

## Low-level

### Data flow through one request

1. `POST /extract {"schema_name": "invoice_v1", "text": "..."}` arrives at `extraction_api.py`.
2. `extraction_normalization.normalize_text(text)` — collapses whitespace (reuses
   `optimization.prompt_compression`'s internal whitespace helpers as a building block),
   strips control characters, enforces a max length (`limits.max_prompt_tokens`-derived char
   cap, rejecting with a 413 rather than silently truncating).
3. `extraction_prompts.get_prompt(schema_name)` resolves `schema_name` to a registered Pydantic
   schema class (`invoice_v1` -> `InvoiceExtraction`, `support_ticket_v1` -> the new
   `SupportTicketExtraction`) and a `prompt_version` string (curriculum's own trace-model field,
   Module 21 §2) — unknown `schema_name` returns 404 before any LLM call.
4. `ExtractionPipeline(runtime, schema, max_repair_attempts=2).run(normalized_text, model)` runs
   the reused Module 8 pipeline. The runtime call inside it is wrapped with `with_retries(...,
   retryable=(RuntimeUnavailable, RequestTimeout))` — transport failures get exponential-backoff
   retry; validation failures do not (they already get the pipeline's own semantic repair retry,
   a different mechanism for a different failure class, per Module 6's own retry-taxonomy rule).
5. `extraction_storage.ExtractionStore.save(ExtractionRecord(...))` persists the full result —
   real SQLite, real file, survives process restart.
6. Response assembled: `request_id`, `status` (`"success"`/`"needs_review"`/`"failed"`), `data`
   (the validated fields), `confidence`, `validation_errors`, `latency_ms` (real wall-clock,
   measured with `runtimes.base.Timer`, the same primitive every other module's latency
   measurement already uses).

### Storage schema (`extraction_storage.py`)

```sql
CREATE TABLE extractions (
    request_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    raw_input TEXT NOT NULL,
    extracted_fields TEXT NOT NULL,   -- JSON-encoded dict
    confidence TEXT NOT NULL,          -- "low" | "medium" | "high"
    needs_review INTEGER NOT NULL,     -- 0 | 1
    validation_error TEXT,
    used_repair_retry INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_extractions_needs_review ON extractions (needs_review);
```

`list_low_confidence()` is `SELECT * FROM extractions WHERE needs_review = 1 ORDER BY
created_at DESC` — deliberately keyed on the pipeline's own `needs_review` flag (which already
folds in *both* "confidence == low" *and* "validation ultimately failed"), not a raw
`confidence = 'low'` string match, so a request that failed validation entirely still shows up
for review even in the rare case its heuristic confidence score wasn't computed as "low."

### API contract

| Endpoint | Method | Request | Response | Errors |
|---|---|---|---|---|
| `/extract` | POST | `{"schema_name": str, "text": str}` | `{"request_id", "status", "data", "confidence", "validation_errors", "latency_ms"}` | 404 unknown schema, 413 text too long, 429 admission queue full (Module 6.5's `AdmissionController`, reused), 504 on `RequestTimeout` after retries exhausted |
| `/extractions/{request_id}` | GET | — | full stored `ExtractionRecord` | 404 not found |
| `/extractions/low-confidence` | GET | `?limit=50` | list of stored records with `needs_review=1` | — |
| `/health`, `/ready` | GET | — | Module 23's existing checks, unchanged | — |

### Error handling

Every error class already exists in `runtimes/errors.py` (Module 6's taxonomy) — this project
adds no new exception types, only new HTTP-status mappings for FastAPI:

| Internal error | HTTP status |
|---|---|
| Unknown `schema_name` | 404 |
| Normalized text exceeds length cap | 413 |
| `QueueFullError` (Module 6.5) | 429 |
| `RequestTimeout` after `with_retries` exhausts attempts | 504 |
| `RuntimeUnavailable` after `with_retries` exhausts attempts | 503 |

### Confidence and evidence in the response

`confidence` is `ExtractionResult.confidence` verbatim (Module 8's `ConfidenceLevel`).
"Evidence" (curriculum's own field name in the API sketch) is `ExtractionResult.fields` itself
for schemas that carry an `evidence: dict[str, str]` field (`InvoiceExtraction` does, per-field
supporting text spans); `SupportTicketExtraction` does not carry a separate evidence field —
documented as a deliberate scope choice in REPORT.md, not a silent gap.

### Two schemas

1. `invoice_v1` -> `local_ai_core.extraction.schemas.InvoiceExtraction` (reused verbatim).
2. `support_ticket_v1` -> `schemas/support_ticket_schema.py`'s new `SupportTicketExtraction`
   (`category: Literal["account","billing","technical","security"]` — reusing Module 19's exact
   four-category taxonomy for continuity with the rest of this course's Nimbus theme,
   `urgency: Literal["low","medium","high"]`, `mentioned_product: str | None`,
   `customer_email: str | None`, `summary: str | None`).
