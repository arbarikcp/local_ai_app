# Proposal — Project 1: Local Structured Extraction Service

> Bible reference: [curriculum.md §34](../../curriculum.md#34-project-1--local-structured-extraction-service) · Structure convention: [projects/PROJECT_TEMPLATE.md](../PROJECT_TEMPLATE.md)

## Why

Structured extraction — turning a free-text document into validated, typed fields — is the
single most common production use case for a local LLM in an enterprise setting: invoices,
support tickets, HR forms, legal clauses. It's also the use case this course has been building
toward piece by piece since Module 1 (JSON output), Module 8 (schema validation, repair,
confidence scoring), and Module 23 (a real deployable service). Project 1 is where those pieces
become one running thing: a service a real operator could point real documents at, get back
validated structured data with a confidence signal, and query for the extractions that need a
human look.

This is also the first Project in the curriculum, closing the loop from "build a component"
(every module 1-23) to "build a product" (every project). Getting its structure right —
proposal, architecture, code, how-to-run, retrospective, all written in that order — sets the
convention every later project follows.

## How

**Reused, not rebuilt** (this course's established discipline since Module 19's QLoRA
precedent):

- `local_ai_core/extraction/pipeline.py`'s `ExtractionPipeline` — prompt building, constrained
  decoding request, JSON parsing, Pydantic validation, bounded repair retry, confidence
  computation. This is the extraction engine; Project 1 doesn't touch its internals.
- `local_ai_core/extraction/schemas.py`'s `InvoiceExtraction` — one of this project's two
  schemas, verbatim.
- `local_ai_core/extraction/confidence.py`'s `compute_confidence()`, `local_ai_core/extraction/json_parsing.py`'s
  `try_parse_json()`, `local_ai_core/extraction/chunking.py`'s `chunk_text()`/
  `merge_partial_extractions()` — all reused inside `ExtractionPipeline`, none reimplemented.
- `local_ai_core/deployment/app_context.py`'s `AppContext`/`build_app_context()` — Module 23's
  composition root, extended (not replaced) with an extraction-specific data-dir entry and
  storage handle.
- `local_ai_core/deployment/health.py`, `backup.py` — startup checks and backup/restore, reused
  unchanged against the new extraction database.
- `local_ai_core/runtimes/base.py`'s `with_retries()` — transport-level timeout/unavailability
  retry, composed around (not instead of) the pipeline's own semantic repair retry.

**Built fresh** (confirmed, by survey, that nothing in the repo already does this):

- `app/storage.py` — persistent SQLite storage for every extraction request (raw input,
  extracted output, validation status, errors, confidence, evidence), plus the one query
  pattern that doesn't exist anywhere yet in this repo: "list extractions below a confidence
  threshold."
- `app/normalization.py` — text normalization before extraction (whitespace, encoding); nothing
  purpose-built exists in the repo, `optimization/prompt_compression.py`'s whitespace helpers
  are reused as a building block only.
- `schemas/support_ticket.py` — a second extraction schema (support tickets — curriculum's own
  suggested use case, continuing this course's running Nimbus Cloud Storage theme), since the
  repo's only two existing schemas are `InvoiceExtraction` and the toy `PersonExtraction`.
- `app/api.py` — the `POST /extract` endpoint and the low-confidence review endpoint, following
  Module 23's `api_app.py` FastAPI pattern exactly (lazy `AppContext` via `get_ctx()`).
- `evals/` — an evaluation command against a labeled dataset covering both schemas (the repo's
  existing 6-record golden set only cleanly covers `PersonExtraction`; this project extends it).

## What this achieves

A running FastAPI service (`app/api.py`) that:

1. Accepts a document as JSON text (file upload is a documented, tested extension point, not
   separately implemented — see ARCHITECTURE.md's low-level section for why).
2. Extracts against one of two schemas (`invoice_v1`, `support_ticket_v1`).
3. Validates every output with Pydantic; repairs an invalid output at most twice
   (curriculum's own cap, bumped from Module 8's default of 1 for this project specifically).
4. Persists raw input, extracted output, validation status, errors, confidence, and evidence to
   a real SQLite database under `~/.local-llm-ai/extraction/`.
5. Returns confidence and evidence in every response.
6. Exposes `GET /extractions/low-confidence` to list extractions needing review.
7. Ships `evals/run_eval.py`, a command that scores the service against a labeled dataset and
   reports curriculum's own metric set.

## How success is measured

Curriculum's own evaluation metrics, computed for real wherever this dev machine allows
(everything except real model quality, which needs a real model this machine doesn't run):

| Metric | How Project 1 measures it | Honest-skip status |
|---|---|---|
| Exact match per field | `evals/metrics.py::field_exact_match()` against the labeled dataset | Real — deterministic comparison, no model needed |
| Missing field rate | Same harness — fraction of required fields returned `null` | Real |
| Hallucinated field rate | Same harness — fraction of fields present but not supported by `evidence` | Real (structural check), honest-skip for real semantic hallucination against a real model |
| Invalid JSON rate | `ExtractionPipeline`'s own `validation_error`/`used_repair_retry` fields, aggregated | Real — exercised via `FakeRuntime` scripted to emit invalid JSON |
| Schema validation failure rate | Same aggregation | Real |
| Latency | `latency_ms` on every stored extraction, real wall-clock time even against `FakeRuntime` | Real (timing is real; absolute numbers won't match a real model until run on the resourced Mac) |
| Manual quality score | Not automatable — documented in REPORT.md as deferred to a human reviewer using `GET /extractions/low-confidence` on the resourced Mac | Honest-skip |

A metric only counts as "measured" in REPORT.md if it has a real, printed number from a real run
of the evaluation command — not a claim.
