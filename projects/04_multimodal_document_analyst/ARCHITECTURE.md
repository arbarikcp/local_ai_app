# Architecture — Project 4: Multimodal Document Analyst

> See [PROPOSAL.md](PROPOSAL.md) for why this exists and how success is measured.

## High-level

```text
Input PDF
  -> load_pdf_document()                                         [reused, Module 9/18]
  -> per page:
       -> screen_document_for_ingestion()                         [reused, Module 22]
       -> doc_routing.decide_route()                               [new, composes reused pieces]
            -> should_use_vlm(text_layer)                          [reused, Module 18]
            -> estimate_image_tokens() + estimate_context_budget_impact()  [reused, Module 18]
       -> route == TEXT_LLM:
            -> doc_extraction.extract_fields() (ExtractionPipeline) [reused, Module 8/Project 1]
       -> route == VLM:
            -> render_page_to_image() -> VisionLanguageModel.describe() [reused, Module 18]
       -> doc_storage.save_page_analysis()                          [new]

Question answering:
  doc_qa.answer_question()                                          [new]
    -> per-page text (already extracted/described)
    -> build_rag_prompt()-style context packing with page citations   [reused pattern, Module 11]
    -> runtime.generate()                                            [reused, Module 6]
    -> citations_are_grounded() per page citation                    [reused, Module 13/Project 2]
```

**Deployment shape**: a single FastAPI process (Module 23's `AppContext` pattern extended,
exactly like Projects 1 and 2), backed by one new SQLite database
(`~/.local-llm-ai/multimodal/multimodal.db`). No new vector store or embedding model needed —
Q&A here works over a small, already-fully-extracted per-page text set, not a large corpus
requiring retrieval (a real, documented scope distinction from Project 2's RAG service).

**Reused components, exact source**:

| Component | Source | Role here |
|---|---|---|
| `render_page_to_image`, `extract_text_layer`, `extract_layout`, `extract_tables` | `local_ai_core/multimodal/pdf_extraction.py` | per-page rendering and text/layout extraction |
| `should_use_vlm`, `MultimodalRoute`, `RoutingDecision` | `local_ai_core/multimodal/routing.py` | the base per-page routing signal |
| `estimate_image_tokens`, `estimate_context_budget_impact` | `local_ai_core/multimodal/memory_cost.py` | real image-token cost, now wired into a decision |
| `VisionLanguageModel`, `FakeVLM` | `local_ai_core/multimodal/vlm.py` | the one honest-skip model call |
| `load_pdf_document` | `local_ai_rag/loaders/pdf_loader.py` | per-page `Document` loading, page-encoded `doc_id` |
| `ExtractionPipeline` | `local_ai_core/extraction/pipeline.py` | structured field extraction (Project 1's engine, reused generically) |
| `citations_are_grounded` | `local_ai_core/evals/citation_verifier.py` | page-citation correctness (Project 2's pattern, reused unchanged) |
| `screen_document_for_ingestion` | `local_ai_core/security/rag_ingestion_guard.py` | per-page injection screening (Module 22, already proven wired into Project 2) |
| `AppContext`, `build_app_context` | `local_ai_core/deployment/app_context.py` | composition root |
| FastAPI `get_ctx()` lazy-context pattern | `scripts/module_23/api_app.py` (via Projects 1/2's own API files) | copied exactly |

**New components** (why nothing existing covers them — see PROPOSAL.md's survey): the routing
gap closure, document/form schema, persistent per-page storage, page-cited Q&A, the FastAPI
layer, and a genuinely multi-page fixture.

## Low-level

### The fixture (`datasets/multimodal/project_04/multi_page_form.pdf`)

A real, 3-page "Account Closure Request" form, built the same way Module 18 built its own
fixtures (`fitz`/PyMuPDF, committed once, regenerable via `scripts/build_fixture.py`):

- **Page 1** (digital-native text): applicant name, account ID, request date.
- **Page 2** (digital-native text): reason for closure (a real paragraph), refund amount owed.
- **Page 3** (scanned, no text layer): a rendered "signature and ID verification" image, real
  embedded PNG, deliberately no `insert_text()` call — the same honest OCR-gap construction
  Module 18's `scanned_receipt.pdf` uses.

`load_pdf_document()` yields `doc_id`s `multi_page_form::page1`, `::page2`, `::page3` — pages 1-2
route to `TEXT_LLM`, page 3 routes to `VLM`, entirely from `should_use_vlm()`'s real text-layer
check, not hardcoded per-page logic.

### Routing (`doc_routing.py`)

```python
@dataclass(frozen=True)
class PageRoutingDecision:
    route: MultimodalRoute
    reason: str
    image_tokens: int | None          # only computed when route == VLM
    context_budget_fraction: float | None

def decide_route(text_layer: str, *, image: Image.Image | None, context_window: int) -> PageRoutingDecision
```

Calls `should_use_vlm(text_layer)` first (Module 18's real decision); if it routes to `VLM`,
additionally computes `estimate_image_tokens(image.width, image.height)` and
`estimate_context_budget_impact(...)` against the caller's `context_window` — real numbers
attached to the routing decision, not just a route label. This is the confirmed gap this project
closes: `memory_cost.py`'s math existed but was never wired into an actual decision anywhere in
the repo before this.

### Document/form schema (`schemas/doc_schemas.py`)

```python
class DocumentFieldExtraction(BaseModel):
    document_type: str | None = None
    applicant_name: str | None = None
    key_date: str | None = None
    key_amount: float | None = None
    notes: str | None = None
    confidence: Literal["low", "medium", "high"]
    evidence: dict[str, str] = {}
```

Mirrors `InvoiceExtraction`'s shape (a required, model-self-reported `confidence`/`evidence`
pair — the same Module 8 schema-requirement Project 1's REPORT.md already documented and worked
around) but generalized for form-shaped documents rather than invoices specifically.

### Storage schema (`doc_storage.py`)

```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,          -- the PDF stem, e.g. "multi_page_form"
    source_path TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    status TEXT NOT NULL,             -- "ingested" | "failed"
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE page_analyses (
    page_id TEXT PRIMARY KEY,         -- e.g. "multi_page_form::page1"
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    route TEXT NOT NULL,              -- "text_llm" | "vlm"
    route_reason TEXT NOT NULL,
    extracted_text TEXT NOT NULL,     -- text-layer text, or VLM description
    extracted_fields TEXT,            -- JSON-encoded dict, text_llm route only
    confidence TEXT,
    needs_review INTEGER,
    quarantine_reason TEXT
);
```

Same idiom as every prior project's store: stdlib `sqlite3`, `check_same_thread=False` (applied
from the start, Project 1's own real bug fix), frozen-dataclass records.

### Q&A with page citations (`doc_qa.py`)

```python
@dataclass(frozen=True)
class PageCitation:
    page_id: str            # "multi_page_form::page2"
    page_number: int
    verified: bool

@dataclass(frozen=True)
class DocQaResult:
    answer: str
    citations: list[PageCitation]
    latency_ms: float

async def answer_question(pages: list[PageAnalysisRecord], question: str, runtime: LLMRuntime, model: str) -> DocQaResult
```

Builds a context block from every page's `extracted_text` (text-layer or VLM description,
whichever route produced it), tagged `[page_id]` per page, the same convention
`citation_packer.build_context()` uses for chunks. **Real finding, not an assumption**:
`citation_packer.extract_citations()`'s regex requires a citation marker to end in
`::<digits>` (a chunk index) - confirmed it does NOT match a bare page id like
`multi_page_form::page2` (`extract_citations("... [multi_page_form::page2].")` returns `[]`).
Page ids here have no trailing chunk index, so `doc_prompts.extract_page_citations()` is a new,
page-id-shaped regex (`prompts/doc_prompts.py`), not a reuse of `citation_packer`'s extractor.
`citations_are_grounded([citation], [p.page_id for p in pages])` (Project 2, reused unchanged) is
still fully reusable downstream - it only does set-membership, confirmed id-format-agnostic.

### API contract

| Endpoint | Method | Request | Response | Errors |
|---|---|---|---|---|
| `/documents` | POST | `{"source_path": str}` | list of per-page `{"page_id", "route", "status"}` | 404 file not found, 422 not a PDF |
| `/documents/{doc_id}` | GET | — | stored `DocumentRecord` + all `page_analyses` | 404 |
| `/documents/{doc_id}/extract` | POST | `{"page_number": int?}` | structured fields per page (all pages if omitted) | 404 doc/page not found |
| `/documents/{doc_id}/query` | POST | `{"question": str}` | `{"answer", "citations": [{"page_id","page_number","verified"}]}` | 404, 429/503/504 (Module 6.5/6, reused) |
| `/health`, `/ready` | GET | — | Module 23's existing checks, unchanged | — |

### Error handling

No new exception types — same discipline as every prior project:

| Internal error | HTTP status |
|---|---|
| Source file not found | 404 |
| Not a `.pdf` file | 422 |
| Unknown `doc_id`/page | 404 |
| `QueueFullError` (Module 6.5) | 429 |
| `RequestTimeout`/`RuntimeUnavailable` after retry (Module 6) | 504 / 503 |

A quarantined page (injection pattern matched) is **not** an error — same framing Project 2
established: the request succeeded, that page didn't pass screening, recorded with a
`quarantine_reason`, excluded from extraction/Q&A context.
