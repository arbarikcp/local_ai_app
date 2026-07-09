# Module 10 — Vector Search and Local Vector Databases

> Phase: Application primitives · Bible reference: [curriculum.md §20](../../curriculum.md#20-module-10--vector-search-and-local-vector-databases)

## Goal

Learn vector storage options and production trade-offs — three real backends behind one
`VectorStore` interface, so Module 11+ retrieval code can swap backends without a rewrite.

> **Machine note:** unlike Module 9's embedding *model* adapters, Chroma and LanceDB are
> vector-database *libraries*, not LLM runtimes or model weights — installing them does not
> violate this repo's no-LLM-runtime constraint ([[project-local-ai-app-curriculum]]). Both are
> installed (`uv add chromadb lancedb`) and every lab in this module runs for real against
> them. `onnxruntime<1.20` is pinned alongside `chromadb` because the latest `onnxruntime`
> only ships wheels for macOS 14+; this machine runs macOS 13.

## 1. Brute-force search

Module 9's `NumpyEmbeddingIndex` — score every stored vector against the query vector, sort,
take the top k. Correct, simple, and O(n) per query: the reference every other backend in
this module is checked against, wrapped here as `NumpyVectorStore` to satisfy the same
`VectorStore` protocol as the real backends.

## 2. ANN search

Approximate Nearest Neighbor search (e.g. HNSW, the index Chroma builds by default) trades
exact top-k for sublinear query time by not comparing against every vector — at the cost of
occasionally missing the true nearest neighbor. Chroma and LanceDB both build ANN indexes
internally; this module doesn't reimplement HNSW from scratch (out of scope — a nontrivial
graph-index algorithm, not a few functions), but §"Real proof" below measures recall@k against
the brute-force reference to make the accuracy trade-off visible rather than assumed.

## 3. Indexing

Chroma's HNSW index is built automatically as vectors are added; the collection is created
with `metadata={"hnsw:space": "cosine"}` so its internal distance metric matches the cosine
similarity this course uses everywhere else (Chroma's default is squared L2). LanceDB's
`.search(vector).metric("cosine")` makes the equivalent choice per-query rather than
per-index.

## 4. Metadata filters

Every `VectorStore.search()` implementation accepts the same `metadata_filter: dict` shape
Module 9's `NumpyEmbeddingIndex` established (exact-match, all keys must match) — see
`vector_store.py`. The three backends satisfy it three different ways:

- `NumpyVectorStore`: Python dict comparison over every candidate (brute-force, same as
  Module 9).
- `ChromaVectorStore`: pushed down to Chroma's own `where` clause (`$and` of exact-match terms
  for multi-key filters — Chroma rejects a bare multi-key dict).
- `LanceDBVectorStore`: metadata is stored as a JSON string column (Arrow/LanceDB have no
  free-form-dict column type), so filtering happens **client-side** after a full-table vector
  search rather than pushed down as SQL — correct, but not what a schema-per-field design
  would allow. Documented as a real trade-off, not silently glossed over.

## 5. Hybrid search

`hybrid.py`'s `hybrid_search()` combines vector search with a keyword/term-overlap signal
using Reciprocal Rank Fusion (RRF) — see §"From-scratch implementation: hybrid search" below.

## 6. Persistence

`ChromaVectorStore(collection_name, path=...)` and `LanceDBVectorStore(table_name, path=...)`
both persist to disk when given a path (`PersistentClient`/`lancedb.connect(path)`) instead of
the in-memory default used for fast tests. §"Real proof" demonstrates persistence across a
fresh client instance pointed at the same path — an actual restart, not a mock, the same
proof standard Module 8.5's `SessionStore` used for SQLite.

## 7. Incremental updates

All three `VectorStore.add()` implementations use upsert semantics: adding with a doc_id
already in the store **overwrites** it, rather than creating a duplicate. This is not
automatic — Chroma's plain `add()` silently *keeps the original* document on a duplicate id
(discovered while testing this module, not assumed from the docs), and LanceDB's plain `add()`
*appends a duplicate row*. Both stores use their real upsert primitives instead
(`collection.upsert()`, `table.merge_insert(...).when_matched_update_all()...`).

## 8. Deletes and reindexing

`VectorStore.delete(doc_id)` is part of the protocol, not an afterthought bolted on later.
`NumpyEmbeddingIndex.delete()` was added retroactively to Module 9's class specifically to
support this (a no-op on a missing id, consistent with the other two backends). "Reindexing"
in this module's scope means: delete the stale vector, add the new one — a full corpus
re-embed-and-reload is Module 11+'s ingestion-pipeline concern, not implemented here.

## 9. Local vector DBs

| Option | Good for | Trade-off | Used here |
|---|---|---|---|
| NumPy/FAISS-style simple store | learning internals | not production enough | `NumpyVectorStore` (Module 9's `NumpyEmbeddingIndex`) |
| SQLite + vector extension | small embedded apps | extension complexity | not implemented — `sqlite-vec`/`sqlite-vss` are separate native extensions this module doesn't add |
| Chroma | quick local RAG | abstraction may hide details | `ChromaVectorStore` |
| LanceDB | embedded production-style vector search | must learn data model/indexing | `LanceDBVectorStore` |
| DuckDB + Parquet + vectors | analytical metadata-heavy workloads | vector support may require custom setup | not implemented — out of scope for this module's three-way comparison |

## 10. Operational trade-offs

Real, measured in §"Real proof" below rather than assumed from the table: search latency and
recall@k for the same corpus, same queries, across all three backends. The headline trade-off
this course cares about (curriculum's own framing): **brute-force NumPy is exact but doesn't
scale; Chroma and LanceDB trade a small amount of exactness (via ANN indexing) for
production-shaped features (persistence, metadata pushdown, a real query engine) — but neither
replaces metadata/ACL filtering, which must happen regardless of backend.**

## Metadata-first retrieval

Production RAG is never just `query -> vector top-k`. It's:

```text
query
  -> auth/ACL filter
  -> metadata filter
  -> vector/hybrid retrieval
  -> rerank
  -> context pack
```

Every `VectorStore.search()` in this module accepts `metadata_filter` as a first-class
parameter (not bolted on after retrieval) for exactly this reason — `tenant_id`, `source_type`,
`security_level`, and `language`-style filters are the curriculum's own named examples of
production filter fields. Reranking and context packing are Module 12's subject, not this
module's.

## From-scratch implementation: hybrid search

```text
query -> keyword_search(query, documents)   -> ranked doc_ids (term overlap)
query -> store.search(query_embedding)      -> ranked doc_ids (cosine similarity)
                                             -> reciprocal_rank_fusion(both rankings)
                                             -> top-k SearchResults
```

`hybrid.py`'s `keyword_score()` is a simple term-overlap ratio — a deliberately crude, real
signal in the same spirit as Module 9's `FakeEmbedder`, not a BM25 implementation (BM25 adds
term-frequency saturation and inverse-document-frequency weighting this module doesn't
implement). The two rankings are combined with **Reciprocal Rank Fusion**, not a weighted sum
of raw scores: cosine similarity (bounded [-1, 1]) and a term-overlap ratio aren't on
comparable scales, but rank position always is — `score(doc) = sum over rankings of 1 / (60 +
rank)`, the standard RRF constant.

## Vector store comparison

| Store | Backend reality | Metadata filter push-down | Persistence |
|---|---|---|---|
| `NumpyVectorStore` | pure Python/NumPy, in-process | client-side dict comparison | none (in-memory only) |
| `ChromaVectorStore` | HNSW index, cosine space configured explicitly | real `where` push-down (`$and` for multi-key) | `PersistentClient(path=...)` |
| `LanceDBVectorStore` | Lance columnar format, cosine metric per-query | client-side JSON filter (documented trade-off, §4) | `lancedb.connect(path)` (always on-disk) |

## Hands-on labs

1. **Store chunks in Chroma** — `scripts/module_10/store_comparison.py`, real, no honest-skip
   (Chroma is a library, not an LLM runtime).
2. **Store chunks in LanceDB** — same script.
3. **Implement metadata filters** — all three `VectorStore.search(..., metadata_filter=...)`
   implementations; exercised for real in the comparison script and its tests.
4. **Add hybrid search** — `hybrid.py`, exercised in `scripts/module_10/store_comparison.py`.
5. **Benchmark retrieval latency** — `scripts/module_10/benchmark_and_evaluate.py`, real
   `time.perf_counter()` measurements across all three backends on this machine.
6. **Evaluate recall** — same script, reusing Module 9's `eval.py` metric functions
   (`recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`) against each backend.

## Gotchas

- Vector top-k without ACL and metadata filtering is not production retrieval — every
  `VectorStore.search()` signature bakes `metadata_filter` in for this reason.
- Deletes and updates are harder than first-time indexing: **discovered directly while
  building this module** — Chroma's `add()` silently keeps the original document on a
  duplicate id instead of erroring or overwriting, and LanceDB's `add()` silently appends a
  duplicate row. Both `ChromaVectorStore.add()` and `LanceDBVectorStore.add()` use real upsert
  primitives (`upsert()`, `merge_insert(...)`) instead, specifically because of this.
- Embedding model changes invalidate stored vectors — same caution as Module 9's "Embedding
  drift" (§10); not re-solved here, just still true.
- High-dimensional vectors increase storage and memory bandwidth cost — relevant on the
  RAM-constrained Macs this course targets; Module 9's Matryoshka truncation is one lever.
- Approximate indexes improve speed but need recall measurement — §"Real proof" measures it
  rather than asserting Chroma/LanceDB's ANN search is "close enough."

## Deliverable

```text
packages/local_ai_rag/stores/
  vector_store.py
  numpy_store.py
  chroma_store.py
  lancedb_store.py
  hybrid.py
  tests/
scripts/module_10/
  store_comparison.py
  benchmark_and_evaluate.py
reports/module_10_vector_store_comparison_report.md
```
