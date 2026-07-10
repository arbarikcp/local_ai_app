# Architecture — Project 2: Production Local RAG Service

> See [PROPOSAL.md](PROPOSAL.md) for why this exists and how success is measured.

## High-level

```text
Document ingestion:
  POST /documents {source_type, source_path_or_text, doc_id?}
    -> rag_ingestion_service.ingest_document()
         -> loader (markdown_loader | rag_text_loader | pdf_loader)   [reused + 1 new]
         -> screen_document_for_ingestion()                          [reused, newly wired]
         -> content_hash() diff against rag_metadata_store            [reused hash fn, new store]
         -> chunk_document() (configurable strategy)                  [reused]
         -> embedder.embed_documents()                                [reused, FakeEmbedder default]
         -> vector_store.add() (LanceDBVectorStore)                   [reused]
         -> rag_metadata_store.save()                                 [new]

Question answering:
  POST /query {question, k?, rewrite?, metadata_filter?}
    -> rag_query_service.answer_question()
         -> ProductionRagPipeline.answer()                            [reused, entire pipeline]
              -> rewrite_query() (optional)                           [reused]
              -> NaiveRetriever.retrieve()                            [reused]
              -> ACLPredicate filter (metadata_filter)                [reused]
              -> HeuristicReranker.rerank()                           [reused]
              -> pack_context() + order_for_generation()              [reused]
              -> build_rag_prompt() -> runtime.generate()             [reused]
              -> extract_citations()                                  [reused]
         -> citations_are_grounded() + citation_faithfulness_score()  [reused, now gates response]
         -> rag_metadata_store.log_query()                            [new]
```

**Deployment shape**: one FastAPI process (Module 23's `AppContext` pattern extended, exactly
like Project 1), backed by two new persistent stores — a `LanceDBVectorStore` table and a
`rag_metadata_store.db` SQLite database — both under
`~/.local-llm-ai/rag/`. No new deployment mode.

**Reused components, exact source**:

| Component | Source | Role here |
|---|---|---|
| `Document`, loaders | `local_ai_rag/loaders/{markdown,pdf}_loader.py` | document ingestion input shape |
| `Chunk`, `chunk_document()` | `local_ai_rag/chunkers/document_chunker.py` | configurable chunking |
| `FakeEmbedder` | `local_ai_rag/embeddings/fake.py` | default embedder (honest-skip) |
| `LanceDBVectorStore` | `local_ai_rag/stores/lancedb_store.py` | real, persistent vector storage |
| `HeuristicReranker` | `local_ai_rag/rerankers/heuristic_reranker.py` | real reranking |
| `ProductionRagPipeline`, `ProductionRagAnswer`, `TraceLog` | `local_ai_rag/production_pipeline.py` | the entire question-side pipeline |
| `screen_document_for_ingestion()`, `SourceTrust` | `local_ai_core/security/rag_ingestion_guard.py` | injection screening at ingest time |
| `content_hash()` | `local_ai_rag/incremental_indexer.py` | change detection for document updates |
| `recall_at_k`, `precision_at_k` | `local_ai_core/evals/retrieval_metrics.py` | eval |
| `citations_are_grounded`, `citation_faithfulness_score` | `local_ai_core/evals/citation_verifier.py` | citation verification, eval |
| `keyword_overlap_relevance`, `refusal_check` | `local_ai_core/evals/answer_metrics.py` | eval |
| `GoldenCase`, `load_golden_set` | `local_ai_core/evals/golden_set.py` | eval dataset shape |
| `AppContext`, `build_app_context` | `local_ai_core/deployment/app_context.py` | composition root |
| FastAPI `get_ctx()` lazy-context pattern | `scripts/module_23/api_app.py` (via Project 1's `extraction_api.py`) | copied exactly |

**New components** (why nothing existing covers them — see PROPOSAL.md's survey): persistent
document metadata/ingestion-status store, a `.txt` loader, the ingestion-guard wiring, the
FastAPI layer, the memory eval metric, citation verification as a response-gate.

## Low-level

### Data flow through ingestion (`POST /documents`)

1. Request carries either `source_path` (server-local file, for markdown/PDF) or `text` +
   `source_type: "text"` (inline text, for the new `.txt`-equivalent path) plus an optional
   caller-supplied `doc_id`.
2. The matching loader produces one or more `Document` objects (a PDF may yield several,
   one per page, `doc_id` suffixed `::pageN` — unchanged from Module 18's convention).
3. Each `Document.text` is screened via `screen_document_for_ingestion(text,
   source_trust=SourceTrust.UNTRUSTED)` — **every** upload is treated as untrusted regardless of
   caller identity (no auth system exists in this project; Project 5's inference gateway owns
   that). A quarantined document is recorded in `rag_metadata_store` with `status="quarantined"`
   and never reaches the embedder or vector store — the endpoint still returns 200 (the *request*
   succeeded; the *document* didn't pass screening) with the quarantine reason in the response
   body, not a 4xx (curriculum's own framing: a compromised trusted source is a real threat, not
   a client error).
4. `content_hash(document.text)` is compared against the stored hash for `document.doc_id`
   (`rag_metadata_store.get_document()`). Three outcomes: new doc_id → ingest; existing doc_id,
   same hash → skip re-embedding, return `status="unchanged"`; existing doc_id, different hash →
   delete the old chunks from the vector store first, then re-ingest (Module 12's
   `IncrementalIndexer` update semantics, reused as the *algorithm*, reimplemented against
   persistent storage since `IncrementalIndexer` itself is in-memory only).
5. `chunk_document(document, max_chars=500)` (the default fixed-size strategy;
   `chunking_strategy` is a documented extension point — see OUTRO.md for why parent-child/
   structural chunking aren't wired into the API in v1 despite being real and available).
6. `embedder.embed_documents([c.text for c in chunks])` — `FakeEmbedder` by default.
7. `vector_store.add(chunk_id, chunk.text, embedding, metadata={"doc_id": ..., "title": ...})`
   per chunk.
8. `rag_metadata_store.save(DocumentRecord(doc_id, source_path, title, status="ingested",
   content_hash, chunk_count, ingested_at))`.

### Data flow through a query (`POST /query`)

1. `ProductionRagPipeline.answer(question, rewrite=<bool>, k=<int>)` runs the entire reused
   pipeline (see high-level diagram), optionally filtered by `metadata_filter` via an
   `ACLPredicate` built from the request's filter dict.
2. The verification gate runs `citations_are_grounded(answer.citations, [c.doc_id for c in
   answer.packed_chunks])` and, for each citation, `citation_faithfulness_score()` against the
   packed chunk text. A citation that fails groundedness is **flagged** in the response
   (`citations[].verified: false`), never silently dropped — curriculum's "citation verifier"
   stage is a check the caller can see and act on, not a filter that hides the model's mistake.
3. `rag_metadata_store.log_query(QueryLogRecord(query_id, question, answer_text, citation_count,
   verified_citation_count, latency_ms, created_at))` — every query is logged, mirroring
   Project 1's audit-logging discipline.
4. Response assembled matching curriculum's exact shape (see "API contract" below).

### Storage schema (`rag_metadata_store.py`)

```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT,
    title TEXT,
    status TEXT NOT NULL,          -- "ingested" | "quarantined" | "unchanged" | "failed"
    content_hash TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE query_log (
    query_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    citation_count INTEGER NOT NULL,
    verified_citation_count INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

Same idiom as Project 1's `extraction_storage.py`: stdlib `sqlite3`,
`check_same_thread=False` (Project 1's own real bug fix, applied from the start here rather than
rediscovered), frozen-dataclass records.

### API contract

| Endpoint | Method | Request | Response | Errors |
|---|---|---|---|---|
| `/documents` | POST | `{"source_type": "markdown"\|"text"\|"pdf", "source_path": str?, "text": str?, "doc_id": str?}` | `{"doc_id", "status", "chunk_count", "quarantine_reason": str?}` (one entry per resulting `Document`, e.g. per PDF page) | 400 malformed request, 422 invalid `source_type` |
| `/query` | POST | `{"question": str, "k": int=5, "rewrite": bool=false, "metadata_filter": dict?}` | curriculum's exact shape: `{"answer", "citations": [{"document_id","chunk_id","score","text_preview","verified"}], "trace": {"retrieved_chunks","reranked_chunks","context_tokens","model"}}` | 429 admission queue full, 503/504 runtime unavailable/timeout |
| `/documents/{doc_id}` | GET | — | stored `DocumentRecord` | 404 not found |
| `/documents/{doc_id}` | DELETE | — | `{"deleted": true, "chunks_removed": int}` | 404 not found |
| `/eval/rag` | POST | `{"golden_set_path": str?}` | the full `RagEvalSummary` (see PROPOSAL.md's metrics table) | — |
| `/health`, `/ready` | GET | — | Module 23's existing checks, unchanged | — |

### Error handling

No new exception types — same taxonomy discipline as Project 1:

| Internal error | HTTP status |
|---|---|
| Unknown `doc_id` (GET/DELETE) | 404 |
| Invalid `source_type` | 422 |
| `QueueFullError` (Module 6.5) | 429 |
| `RequestTimeout` after retry | 504 |
| `RuntimeUnavailable` after retry | 503 |

A quarantined document is **not** an error — see ingestion step 3 above.

### Citation response shape, mapped to curriculum's field names

| Curriculum field | This project's source |
|---|---|
| `document_id` | `citation.split("::")[0]` (chunk_id's doc-id prefix) |
| `chunk_id` | the raw citation marker (`extract_citations()`'s output, already `doc_id::index` shaped) |
| `score` | the packed chunk's `SearchResult.score` |
| `text_preview` | first 200 chars of the packed chunk's text |
| `verified` (new field, not in curriculum's literal sketch but required by PROPOSAL.md's citation-verifier-as-gate commitment) | `citations_are_grounded()` result for that specific citation |
