# Proposal — Project 4: Multimodal Document Analyst

> Bible reference: [curriculum.md §37](../../curriculum.md#37-project-4--multimodal-document-analyst) · Structure convention: [projects/PROJECT_TEMPLATE.md](../PROJECT_TEMPLATE.md)

## Why

Multimodal document analysis — scanned forms, invoices, screenshots, diagrams — is curriculum's
fourth central production use case, and the one that most directly combines what Projects 1 and
2 already built: structured extraction (Project 1's `ExtractionPipeline`) and citation-verified
Q&A (Project 2's `citations_are_grounded` pattern), now applied to documents where some pages
have a real text layer and some don't. Module 18 already built and proved every individual
piece — PDF page rendering, text-layer extraction, layout/table extraction, image preprocessing,
a `VisionLanguageModel` protocol, and `should_use_vlm()`'s real routing decision. Project 4 is
where that pipeline becomes a product: ingest a real multi-page document, extract structured
fields or answer questions about it, and prove — for real, not asserted — that pages needing a
VLM and pages that don't are routed correctly and cited correctly.

## How

**Reused, not rebuilt** (confirmed by survey):

- `local_ai_core/multimodal/pdf_extraction.py` — `render_page_to_image()`, `extract_text_layer()`,
  `extract_layout()`, `extract_tables()`. Unchanged.
- `local_ai_core/multimodal/routing.py`'s `should_use_vlm()` — the real per-page text-layer-length
  routing decision. Unchanged.
- `local_ai_core/multimodal/vlm.py`'s `VisionLanguageModel` Protocol and `FakeVLM` — this
  project's honest-skip default for the one real model call, same DI pattern every model-backed
  adapter since Module 6 uses.
- `local_ai_core/multimodal/memory_cost.py`'s `estimate_image_tokens()`/
  `estimate_context_budget_impact()` — real image-token math, confirmed by survey to exist as
  pure functions never wired into a routing decision; this project closes that gap (see "Built
  fresh" below).
- `local_ai_rag/loaders/pdf_loader.py`'s `load_pdf_document()` — real per-page `Document` loading,
  page-encoded `doc_id` (`pdf_stem::pageN`), unchanged since Module 18.
- `local_ai_core/extraction/pipeline.py`'s `ExtractionPipeline` — Project 1's reused structured-
  extraction engine, confirmed generic (no dependency on Project 1's own schemas), applied here
  to a new document-shaped schema.
- `local_ai_core/evals/citation_verifier.py`'s `citations_are_grounded()` — Project 2's citation-
  correctness check, confirmed id-format-agnostic; works unchanged against page-encoded ids.
- `local_ai_core/security/rag_ingestion_guard.py`'s `screen_document_for_ingestion()` — Module 22's
  injection screen, confirmed already wired into Project 2's ingestion path; this project wires
  it into per-page document ingestion the same way, satisfying Module 22's own named threat
  ("indirect prompt injection via document content") with zero new security code.
- `local_ai_core/deployment/app_context.py`'s `AppContext`/`build_app_context()` — Module 23's
  composition root, extended the same way Projects 1-3 extended it.

**Built fresh** (confirmed, by survey, that nothing in the repo already does this):

- `datasets/multimodal/project_04/multi_page_form.pdf` — a new, real, 3-page fixture (two digital-native
  text pages, one genuinely scanned image-only page) — both existing Module 18 fixtures are
  single-page, too small to exercise page-level citations or a real per-document OCR+LLM-vs-VLM
  comparison within one document.
- `app/doc_routing.py` — a real function composing `should_use_vlm()` with
  `memory_cost.py`'s image-token math, closing the confirmed gap: memory cost is real, computed
  code today but was never wired into an actual routing decision anywhere in the repo.
- `schemas/doc_schemas.py` — a new document/form extraction schema; Project 1's own schemas
  (invoice, support ticket) are text-only with no page provenance concept.
- `app/doc_storage.py` — persistent per-page analysis storage (route taken, extracted fields,
  confidence, citations) — nothing in the repo persists multimodal routing/extraction results.
- `app/doc_qa.py` — page-citation-verified Q&A composing the reused pieces above; no existing
  code assembles a multi-page-aware answer with per-page citation grounding.
- `app/doc_api.py` — the FastAPI layer (`POST /documents`, `GET /documents/{id}`,
  `POST /documents/{id}/extract`, `POST /documents/{id}/query`).

## What this achieves

A running FastAPI service (`app/doc_api.py`) that:

1. Accepts a PDF (image input is a documented extension point — see ARCHITECTURE.md).
2. Extracts text per page via PDF text-layer extraction, screening every page's text for
   injection patterns before any downstream step.
3. Routes each page to text-LLM or VLM analysis via `doc_routing.py`'s real, memory-cost-aware
   decision — not just text-layer length, the confirmed gap this project closes.
4. Extracts structured fields per page via the reused `ExtractionPipeline`.
5. Answers questions about the document via `doc_qa.py`, with citations resolved to real page
   numbers.
6. Verifies every citation against which pages were actually analyzed (`verified: bool`,
   Project 2's pattern, reused unchanged).
7. Persists every page's route, extraction, and confidence — queryable after the fact.
8. Compares the OCR+LLM (text-layer) path against the VLM path explicitly, on the one fixture
   page that genuinely needs it (curriculum's functional requirement 7).

## How success is measured

Curriculum's own evaluation metrics, computed for real wherever this dev machine allows:

| Metric | How Project 4 measures it | Honest-skip status |
|---|---|---|
| OCR quality | Real text-layer character count per page vs. a labeled expected count in `evals/doc_golden_set.jsonl` | Real (PDF text-layer extraction, not OCR — this machine has no OCR library, confirmed by survey; the "quality" measured is text-layer extraction fidelity, documented honestly, not conflated with real OCR accuracy) |
| Field extraction accuracy | Reuses Project 1's `field_exact_match()`-style comparison against labeled reference fields | Real |
| Page citation correctness | `citations_are_grounded()` (Project 2, reused) against real page-encoded ids | Real |
| Latency | Real wall-clock `Timer` measurement per document | Real |
| Memory | Reuses Project 2's `psutil`-based peak-RSS pattern | Real |
| Model failure cases | A scripted adversarial VLM/LLM response per curriculum's own failure-case framing (Project 3's precedent), proven caught | Real |
| OCR+LLM vs. VLM comparison | `doc_routing.py`'s decision applied to all 3 fixture pages, with the routed-to path's real output compared side by side | Real (routing is real; VLM *output quality* is `FakeVLM`-backed, honest-skip for real visual reasoning, deferred to the resourced 32GB Mac) |

A metric only counts as "measured" in REPORT.md if it has a real, printed result from a real run
of `evals/run_doc_eval.py` — not a claim.
