# Outro — Project 1: Local Structured Extraction Service

## What this achieved

A real, running FastAPI service that turns free text into validated structured data across two
schemas, persists every attempt with confidence and evidence, exposes a low-confidence review
queue, and ships an evaluation harness that provably distinguishes a flawless run from a broken
one. It's also the first Project in this curriculum — the first time components built across 23
modules (a runtime abstraction, structured-output validation, a composition root, security
guarding, observability) came together as one thing a real operator could point at real
documents. And it caught a real, previously-latent cross-module bug (`AuditLog`'s thread safety)
that 22 prior modules' tests never exercised.

## What's still open (honest-skip, not forgotten)

- **Real model quality.** Every metric in REPORT.md is mechanically real; none of them say
  anything about how well an actual local model extracts real invoices or real support tickets.
  That number only exists once this runs on the resourced 32GB Mac with a real runtime injected
  via `build_extraction_context(..., runtime=<real LLMRuntime>)` — no other code changes needed,
  by design.
- **File upload input.** Curriculum's requirement 1 wants both text and file input; this service
  only accepts JSON text. The natural extension is a `multipart/form-data` variant of `/extract`
  that runs the uploaded bytes through Module 18's `pdf_extraction.py` (for PDFs) before the
  existing `normalize_text()` step — the extraction pipeline itself needs no changes, only a new
  ingress path.
- **PII redaction on stored raw input.** `raw_input` is stored verbatim in `extraction.db`.
  Module 21's `redact_pii()`/`PromptLoggingPolicy` are real and reusable but weren't wired into
  storage here — a real production deployment handling actual customer data would want this
  before go-live, not after.
- **A manual quality score.** PROPOSAL.md committed to this as honest-skip from the start;
  `GET /extractions/low-confidence` is the real, working entry point a human reviewer would use.

## What to explore next

- **Constrained decoding for real**, once this runs against a real MLX/Ollama model — Module 8's
  `ExtractionPipeline` already requests `response_format.type="json_schema"` and falls back
  cleanly on `FeatureNotSupported`; the interesting open question is how much the repair-retry
  rate actually drops with real constrained decoding vs. prompt-only mode. This project's
  evaluation harness is already built to measure exactly that comparison — it just needs a real
  runtime swapped in.
- **A guard-classifier pass on extracted output**, reusing Module 22's `RuleBasedGuardClassifier`
  on `extracted_fields` before persistence — right now a support ticket's `summary` field is
  extracted and stored without any secrets/PII scan, unlike the request path Module 23's `/chat`
  endpoint already guards.
- **Structured output via a real local guard-model-adjacent technique**: grammar-constrained
  decoding (GBNF) is stubbed (`placeholder_gbnf_grammar()`, Module 8) but never exercised against
  a real grammar-capable runtime like llama.cpp — worth a real comparison against
  `json_schema`-mode once on the resourced Mac, since curriculum's own Lab 8 (Module 8) frames
  this as an open three-way comparison this project never got to run for real.
- **Multi-document batch extraction with the existing chunking/merge machinery** —
  `ExtractionPipeline.run_chunked()` and `merge_partial_extractions()` already exist and are
  tested (Module 8) but this project's API only ever calls `.run()` on a single document; a
  `/extract/batch` endpoint processing an uploaded document set would be a natural Project 2-
  adjacent extension once file input (above) exists.
