# Proposal — Project 2: Production Local RAG Service

> Bible reference: [curriculum.md §35](../../curriculum.md#35-project-2--production-local-rag-service) · Structure convention: [projects/PROJECT_TEMPLATE.md](../PROJECT_TEMPLATE.md)

## Why

RAG is this course's other central production use case alongside structured extraction —
answering questions from private documents with citations, not from a model's unverified prior
knowledge. Modules 9-13 built the components (embeddings, vector search, naive RAG, production
retrieval with reranking/hybrid search/query rewriting, evaluation with citation verification)
piece by piece, the same way Modules 1-8 built toward Project 1. Almost nothing here needs
building from scratch — `local_ai_rag`'s `ProductionRagPipeline` already implements most of
curriculum's own query-side architecture diagram. Project 2's job is composing that real,
already-tested machinery into a running service with persistent document metadata, ingestion
status, and an evaluation command — the same "build the product, not the component" step
Project 1 took for extraction.

## How

**Reused, not rebuilt** (confirmed by survey — this is the largest reuse ratio of any project or
module so far):

- `local_ai_rag/loaders/{markdown,pdf}_loader.py` — document loading, unchanged.
- `local_ai_rag/chunkers/document_chunker.py` — configurable-size chunking (curriculum's
  functional requirement 3 is satisfied by this alone; `parent_child_chunker.py` and
  `structural_chunker.py` remain available as alternate strategies, not required for v1).
- `local_ai_rag/embeddings/fake.py`'s `FakeEmbedder` — real bag-of-words embedding (honest-skip
  default; `SentenceTransformersEmbedder`/`OllamaEmbedder` are the documented "enable for real"
  path, same pattern as every model-backed adapter since Module 6).
- `local_ai_rag/stores/lancedb_store.py`'s `LanceDBVectorStore` — a **real**, not honest-skip,
  embedded vector database (curriculum's functional requirement 5, satisfied directly).
- `local_ai_rag/rerankers/heuristic_reranker.py`'s `HeuristicReranker` — real, non-neural
  reranking (curriculum's advanced requirement 2).
- `local_ai_rag/production_pipeline.py`'s `ProductionRagPipeline` — the entire question-side
  pipeline (rewrite → retrieve → ACL filter → rerank → pack context → generate → extract
  citations → trace), reused as-is for the `/query` endpoint's core.
- `local_ai_core/evals/{retrieval_metrics,citation_verifier,answer_metrics,golden_set}.py` —
  recall@k/precision@k, citation groundedness/faithfulness, refusal-based abstention checking,
  golden-set loading — all of curriculum's evaluation metrics except latency and memory already
  have a real implementation to call.
- `local_ai_core/security/rag_ingestion_guard.py`'s `screen_document_for_ingestion()` — built in
  Module 22, real, but never wired into an actual ingestion path until this project (confirmed
  by survey: only called from a standalone demo script).
- `local_ai_core/deployment/app_context.py`'s `AppContext`/`build_app_context()` — Module 23's
  composition root, extended the same way Project 1 extended it (a project-owned data-dir
  subpath, not a change to the shared `DataDirectoryLayout`).

**Built fresh** (confirmed, by survey, that nothing in the repo already does this):

- `app/rag_metadata_store.py` — a persistent SQLite document-metadata + ingestion-status store.
  `IncrementalIndexer` (Module 12) has real diff *logic* but keeps its manifest in memory only;
  nothing survives a process restart today.
- A plain-text (`.txt`) loader — only markdown and PDF loaders exist; curriculum's functional
  requirement 1 explicitly asks for text too.
- `app/rag_ingestion_service.py` — the first real wiring of `screen_document_for_ingestion()`
  into an actual ingest-then-store path.
- `app/rag_api.py` — the FastAPI layer (`POST /documents`, `POST /query`, `GET /documents/{id}`,
  `DELETE /documents/{id}`, `POST /eval/rag`).
- `evals/rag_eval_metrics.py`'s memory metric — no existing eval infra measures memory; the
  seven other evaluation metrics are all real reuse.
- A citation-verifier-as-response-gate — `citations_are_grounded()`/`citation_faithfulness_score()`
  exist as checks a caller can run; nothing today turns them into an API response field the way
  curriculum's architecture diagram's explicit "citation verifier" stage implies.

## What this achieves

A running FastAPI service (`app/rag_api.py`) that:

1. Ingests markdown, `.txt`, and PDF-derived text via `POST /documents`, screening every
   document for injection patterns before it ever reaches the vector store.
2. Persists document metadata and ingestion status (`pending`/`ingested`/`quarantined`/`failed`)
   in real SQLite, survives a process restart.
3. Chunks with a configurable strategy (fixed-size by default, parent-child available).
4. Generates real embeddings (`FakeEmbedder` on this machine, `SentenceTransformersEmbedder`/
   `OllamaEmbedder` swappable via dependency injection).
5. Stores vectors in a real, persistent `LanceDBVectorStore`.
6. Answers questions with citations via `POST /query`, running the full reused
   `ProductionRagPipeline` (rewrite → retrieve → ACL → rerank → pack → generate → cite), plus a
   new citation-verification gate that flags (never silently drops) an ungrounded citation.
7. Supports metadata filters (via `ProductionRagPipeline`'s existing `ACLPredicate` mechanism).
8. Supports document update and delete (`DELETE /documents/{id}`, re-`POST` of a changed
   `doc_id` triggers re-chunk/re-embed via a real content-hash diff).
9. Maintains ingestion status queryable via `GET /documents/{id}`.
10. Ships `POST /eval/rag`, running the evaluation command against a labeled golden set.

## How success is measured

Curriculum's own evaluation metrics, computed for real wherever this dev machine allows:

| Metric | How Project 2 measures it | Honest-skip status |
|---|---|---|
| Recall@k | `local_ai_core.evals.retrieval_metrics.recall_at_k()` against the golden set's `expected_source_ids` | Real |
| Precision@k | Same module's `precision_at_k()` | Real |
| Citation correctness | `local_ai_core.evals.citation_verifier.citations_are_grounded()` | Real |
| Faithfulness | Same module's `citation_faithfulness_score()` (explicitly labeled word-overlap heuristic, not NLI) | Real (heuristic), honest-skip for real semantic faithfulness against a real judge model |
| Answer relevance | `local_ai_core.evals.answer_metrics.keyword_overlap_relevance()` (explicitly labeled heuristic) | Real (heuristic) |
| Abstention accuracy | `local_ai_core.evals.answer_metrics.refusal_check()` against `GoldenCase.requires_refusal` | Real |
| Latency | Real wall-clock time (`Timer`) on every query | Real |
| Memory | `evals/rag_eval_metrics.py`'s new `psutil`-based peak-RSS measurement around the ingestion + eval run | Real (measures this process's real memory; won't reflect a real embedding/generation model's footprint until run on the resourced Mac) |

A metric only counts as "measured" in REPORT.md if it has a real, printed number from a real run
of `POST /eval/rag` (or the equivalent CLI script) — not a claim.
