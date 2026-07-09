# Module 12 — RAG v2: Production Retrieval

> Phase: RAG · Bible reference: [curriculum.md §22](../../curriculum.md#22-module-12--rag-v2-production-retrieval)

## Goal

Evolve naive RAG (Module 11) into production-grade retrieval — every stage of the curriculum's
production pipeline diagram gets real, testable code:

```text
Ingestion
  -> parse document -> normalize text -> split into semantic units -> create chunks
  -> attach metadata -> create embeddings -> persist chunks and vectors -> index metadata

Query
  -> classify query -> rewrite query if needed -> apply ACL/metadata filter
  -> retrieve candidates -> rerank candidates -> select context budget -> pack context
  -> generate answer -> validate citations -> log trace
```

`ProductionRagPipeline` (`production_pipeline.py`) implements the query-side half of this
diagram end to end; `IncrementalIndexer` implements the ingestion-side "persist... index
metadata" step's real-world complement (documents change after first ingestion).

> **Machine note:** same discipline as every module since 9 — real code runs for real wherever
> it doesn't require a live model. Query rewriting, multi-query retrieval, and HyDE all need an
> LLM call (`FakeRuntime` here, real adapter unchanged later). A real cross-encoder reranker
> needs downloaded model weights (`CrossEncoderReranker` uses Module 9's lazy-import/DI
> pattern, honest-skip). Everything else — chunking strategies, parent-child retrieval,
> metadata/ACL filtering, the heuristic reranker, context packing, source-level citations, and
> incremental indexing — runs for real, no honest-skip.

## Scope note: document parsing

The curriculum gives document parsing (PDF layout extraction, OCR, table/header/footer
handling, parser comparison across PyMuPDF/docling/markitdown/unstructured) its own depth
section and a dedicated lab (parse the same table-heavy PDF two ways, compare downstream
accuracy). **Not implemented as code this module** — it needs real messy source documents
(PDFs with tables) and heavy optional dependencies (OCR engines, layout models) whose only
payoff is a single comparison lab, a poor effort/signal trade-off given this repo's markdown-only
corpus (Module 11) and disk-constrained machine. Covered here as theory (§"Document parsing")
plus one concrete piece of code that *is* implemented for real: `structural_chunker.py`'s
table/code-block-aware chunking, which is the exact "chunk boundaries based on document
structure, not only token count" principle the parsing section calls for — demonstrated on
markdown structure (tables, fenced code) rather than PDF layout, but the same principle.

## Core topics

### 1-2. Chunking strategies and semantic chunking

`chunkers/semantic_chunker.py`'s `chunk_semantically()` splits text into sentences, embeds each
with an `Embedder`, and starts a new chunk whenever cosine similarity to the running chunk's
last sentence drops below a threshold — chunk boundaries follow *meaning*, not a fixed
character count. Real algorithm, real embeddings (`FakeEmbedder`), genuinely different chunk
boundaries than Module 11's fixed-size `chunk_text()` on the same input (§"Real proof").

### 3. Parent-child retrieval

`chunkers/parent_child_chunker.py` produces small child chunks (good for precise embedding
matches) each carrying a reference to a larger parent chunk (good for generation context) -
"index small, retrieve big." `retrievers/parent_child_retriever.py`'s
`ParentChildRetriever.retrieve()` searches child embeddings but returns deduplicated parent
text - Lab 1.

### 4. Sliding windows

Already implemented: Module 8's `chunk_text(..., overlap_chars=N)`, reused unchanged by
`document_chunker.py` since Module 11. Not reimplemented - restated here because the curriculum
lists it as a Module 12 topic even though the mechanism predates this module.

### 5-6. Table-aware and code-aware chunking

`chunkers/structural_chunker.py`'s `chunk_preserving_structure()` extracts markdown tables and
fenced code blocks as atomic units *before* running `chunk_text()` on the surrounding prose, then
re-inserts them - both structures use the same "never split an atomic unit" mechanism, not two
separate implementations, since the underlying problem (don't let a size-based splitter cut
through a structured block) is identical for both.

### 7-9. Query rewriting, multi-query retrieval, HyDE

`retrievers/query_expansion.py`:
- `rewrite_query()` — one LLM call to reformulate the question, then retrieve with the rewritten
  text.
- `multi_query_retrieve()` — the LLM generates N query variants; each is retrieved separately;
  results are fused with Module 10's `reciprocal_rank_fusion()` rather than a naive
  score-average, for the same reason hybrid search uses RRF (differently-scaled result sets
  aren't directly comparable, rank position is).
- `hyde_retrieve()` — Hypothetical Document Embeddings: the LLM generates a *hypothetical
  answer* to the question, and that hypothetical answer's embedding is used to search instead
  of the question's embedding (a hypothetical answer's embedding is often closer to real answer
  chunks than the question's own embedding is).

All three need a real LLM to produce good rewrites/hypotheticals - mechanically real, `FakeRuntime`-backed here, quality-honest-skip (§ machine note).

### 10. Hybrid search

Already implemented: Module 10's `hybrid.py`. `ProductionRagPipeline` uses it directly as its
retrieval stage rather than `NaiveRetriever` alone.

### 11. Reranking

`rerankers/heuristic_reranker.py`'s `HeuristicReranker` is a **real** reranker: it recomputes a
combined vector+keyword-overlap score for every candidate and reorders by it - not a neural
cross-encoder, but a genuine, measurable reordering (§"Real proof"). `rerankers/
cross_encoder_reranker.py`'s `CrossEncoderReranker` wraps `sentence-transformers`' `CrossEncoder`
with the lazy-import/DI pattern (Module 9's `SentenceTransformersEmbedder` precedent) -
unit-tested via an injected fake `score_fn`, real model honest-skip.

### 12. Context packing

`context_packers/budget_packer.py` implements the curriculum's exact context-budget shape
(`max_context_tokens`, `reserved_for_system/question/answer`, `available_for_chunks`) and packs
candidate chunks by, in order: relevance score, source diversity (no more than
`max_chunks_per_source` from any one document, so one dominant document can't crowd out
everything else), and token budget - stopping before the running total would exceed
`available_for_chunks`. Uses Module 1's token-estimation heuristic (`token_estimate.py`) for
token counting, not a live tokenizer, consistent with every other module's honest heuristic-vs-
exact distinction.

### 13. Lost-in-the-middle mitigation

`budget_packer.py`'s `order_for_generation()` reorders packed chunks so the highest-relevance
chunks sit at the start and end of the context, not buried in the middle - the empirically
observed weak spot for long-context attention. Documented as a real, applied mitigation, not
just a citation of the phenomenon.

### 14. ACL-aware retrieval

`retrievers/acl.py`'s `AclAwareRetriever` wraps any retriever and applies a caller-supplied
predicate function (`ACLPredicate`, `(metadata) -> bool`) over-fetched candidates *before*
truncating to `k` - real access-control enforcement at the retrieval layer, not left to the
prompt or the generator to "not mention" restricted content. `retrieve()` over-fetches (`k *
fetch_multiplier`) precisely so ACL-filtered-out candidates don't silently shrink the effective
top-k below what the caller asked for.

### 15. Time-aware retrieval

`retrievers/time_aware.py`'s `apply_recency_boost()` combines a candidate's similarity score
with an exponential recency decay based on a `created_at` metadata timestamp - a genuinely
older-but-still-relevant document can still outrank a barely-relevant recent one (decay boosts,
never fully overrides, relevance).

### 16. Incremental indexing

`incremental_indexer.py`'s `IncrementalIndexer` hashes each document's content (SHA-256) and
diffs against a stored manifest: unchanged documents are skipped entirely (no re-embedding
cost), changed documents are re-chunked/re-embedded and their old chunks deleted first, and
documents no longer present are deleted - Lab 6, real and measurable (§"Real proof").

## RAG memory note

A production RAG query can involve three model classes (embedding model, reranker, generator)
that shouldn't all assume simultaneous residency on a memory-constrained Mac (ties to Module 4's
memory math and Module 6.5's serving concurrency). `ProductionRagPipeline` calls each stage
sequentially - embed, then rerank, then generate - never holding all three "models" doing work
at once by construction, though on this machine none of the three is actually a resident model
process (`FakeEmbedder`, `HeuristicReranker`, `FakeRuntime`).

## Context packing strategy

```yaml
max_context_tokens: 6000
reserved_for_system: 500
reserved_for_question: 300
reserved_for_answer: 1000
available_for_chunks: 4200
```

Implemented exactly as `ContextBudget` in `budget_packer.py`, with `available_for_chunks` as a
computed property (`max_context_tokens - reserved_for_system - reserved_for_question -
reserved_for_answer`) rather than a value a caller could set inconsistently with the other
three.

## Hands-on labs

1. **Implement parent-child retrieval** — `chunkers/parent_child_chunker.py` +
   `retrievers/parent_child_retriever.py`, `scripts/module_12/parent_child_demo.py`.
2. **Add metadata filtering** — already real since Module 10; `AclAwareRetriever` extends it to
   predicate-based (non-exact-match) filtering this module.
3. **Add reranking** — `rerankers/heuristic_reranker.py`, `scripts/module_12/reranking_demo.py`.
4. **Add context packing** — `context_packers/budget_packer.py`, same script.
5. **Add source-level citations** — `context_packers/citation_packer.py`'s
   `summarize_source_citations()`, aggregating chunk-level citations (Module 11) to
   deduplicated document-level citations.
6. **Add incremental indexing** — `incremental_indexer.py`,
   `scripts/module_12/incremental_indexing_demo.py`.

## Deliverable

```text
packages/local_ai_rag/
  chunkers/parent_child_chunker.py
  chunkers/semantic_chunker.py
  chunkers/structural_chunker.py
  retrievers/parent_child_retriever.py
  retrievers/query_expansion.py
  retrievers/time_aware.py
  retrievers/acl.py
  rerankers/heuristic_reranker.py
  rerankers/cross_encoder_reranker.py
  context_packers/budget_packer.py
  incremental_indexer.py
  production_pipeline.py
  tests/ (per subpackage)
scripts/module_12/
  parent_child_demo.py
  reranking_demo.py
  incremental_indexing_demo.py
reports/module_12_production_retrieval_report.md
```
