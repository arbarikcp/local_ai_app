# Project 4 — Multimodal Document Analyst

> See [PROPOSAL.md](PROPOSAL.md) for why, [ARCHITECTURE.md](ARCHITECTURE.md) for how it's built,
> [REPORT.md](REPORT.md) for measured results, [OUTRO.md](OUTRO.md) for what's next.

## Setup

No install beyond the repo's own `uv sync` (run once at the repo root). This project adds no new
dependencies — `pymupdf`, `pdfplumber`, `pillow`, and `psutil` are all already real, installed
dependencies from Module 18 and Project 2.

## Run the tests

```bash
uv run pytest projects/04_multimodal_document_analyst -q
```

## Rebuild the fixture (already committed, only needed if you want to regenerate it)

```bash
uv run python projects/04_multimodal_document_analyst/build_fixture.py
```

Writes `datasets/multimodal/project_04/multi_page_form.pdf` — a real, 3-page "Account Closure
Request" form (2 digital-native text pages + 1 genuinely scanned image-only page). Lives in its
own `project_04/` subdirectory, not directly in `datasets/multimodal/`, because
`scripts/module_18/multimodal_rag_demo.py` globs every `*.pdf` directly in that directory
(non-recursive) and hardcodes its own two-fixture assertions — confirmed by a real test failure
this fixture caused there before being moved into its own subdirectory.

## Run the API service

```bash
uv run uvicorn doc_api:app --app-dir projects/04_multimodal_document_analyst/app --port 8000
```

⚠️ With the default config (`config/app.example.yaml`), this writes to the real
`~/.local-llm-ai/multimodal/` on your machine (a small SQLite database) — same disclosed
behavior as every prior project. Set `APP_CONFIG_PATH=/path/to/config.yaml` to point at a
different data directory.

The default VLM is `FakeVLM` (a scripted description, not real visual reasoning) and the default
runtime is `FakeRuntime` (no model runtime installed on this dev machine) — every `/query` call
comes back with `FakeRuntime`'s canned response until a real runtime/VLM is injected via
`build_doc_context(..., runtime=..., vlm=...)`, no other code change needed.

### Example requests

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source_path": "datasets/multimodal/project_04/multi_page_form.pdf"}'

curl http://127.0.0.1:8000/documents/multi_page_form

curl -X POST http://127.0.0.1:8000/documents/multi_page_form/extract \
  -H "Content-Type: application/json" -d '{}'

curl -X POST http://127.0.0.1:8000/documents/multi_page_form/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund amount owed?"}'
```

Real response from the `POST /documents` command above (against the real fixture, showing real
routing including the real memory-cost math the VLM page triggers):

```json
[
  {"page_id": "multi_page_form::page1", "route": "text_llm", "status": "ingested"},
  {"page_id": "multi_page_form::page2", "route": "text_llm", "status": "ingested"},
  {"page_id": "multi_page_form::page3", "route": "vlm", "status": "ingested"}
]
```

`GET /documents/multi_page_form` shows page 3's real routing reason: `"text layer has only 0
chars (< 40 threshold) - likely scanned/image-only; rendered image costs an estimated 2700
tokens (45.0% of a 6000-token context window)"` — the real `estimate_image_tokens()`/
`estimate_context_budget_impact()` numbers this project wires into the routing decision.

## Run the evaluation

```bash
uv run python projects/04_multimodal_document_analyst/evals/run_doc_eval.py
```

Ingests the real, committed 3-page fixture through the real composition root, then scores it
against the real golden set (`evals/doc_golden_set.jsonl`, 3 labeled pages + 3 labeled
questions) against two scripted scenarios — "perfect" (proves the metrics score a flawless run
correctly) and "adversarial" (proves the metrics catch a real, deliberately broken run: wrong
extracted fields, an invented page citation) — proving the metrics harness works correctly, not
claiming real model/VLM quality. See REPORT.md for the actual numbers from a real run, including
a genuine, undoctored gap the adversarial run exposed.

## Available endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health`, `/ready` | GET | Liveness/readiness (Module 23, reused) |
| `/documents` | POST | Ingest a PDF: screen, route, extract-or-describe, persist every page |
| `/documents/{doc_id}` | GET | Retrieve stored document metadata and every page's analysis |
| `/documents/{doc_id}/extract` | POST | Re-run structured extraction for one page or every `text_llm`-routed page |
| `/documents/{doc_id}/query` | POST | Ask a question, get an answer with page citations, each independently verified |
