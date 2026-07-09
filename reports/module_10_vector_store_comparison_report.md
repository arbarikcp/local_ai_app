# Module 10 deliverable — vector store comparison report

Status: **complete.** Unlike most modules, this one has *no* honest-skip surface at all — Chroma
and LanceDB are vector database libraries, not LLM runtimes or model weights, so both are
installed on this machine (`onnxruntime<1.20` pinned alongside `chromadb`, since the latest
`onnxruntime` only ships wheels for macOS 14+ and this machine runs macOS 13) and every lab in
this module runs for real against all three backends.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `packages/local_ai_rag/stores/vector_store.py` | — | `VectorStore` protocol shared by all three backends |
| `packages/local_ai_rag/stores/numpy_store.py` | 8 | Async wrapper around Module 9's `NumpyEmbeddingIndex`, including the `delete()` method added this module |
| `packages/local_ai_rag/stores/chroma_store.py` | 13 | Real Chroma collection: cosine-space search, `upsert`-based overwrite semantics, `where`-clause metadata filtering (`$and` for multi-key), delete, and real on-disk persistence across a fresh client |
| `packages/local_ai_rag/stores/lancedb_store.py` | 13 | Real LanceDB table: cosine-metric search, `merge_insert`-based overwrite semantics, client-side JSON metadata filtering, delete, and real on-disk persistence across a fresh connection |
| `packages/local_ai_rag/stores/hybrid.py` | 12 | Term-overlap keyword scoring, Reciprocal Rank Fusion, and a hybrid search that recovers an exact-code match vector search alone misses |
| `packages/local_ai_rag/embeddings/embedder.py` (`NumpyEmbeddingIndex.delete`) | 3 new | Delete removes the document, is a no-op on a missing id, and the document disappears from search results |
| `scripts/module_10/store_comparison.py` | 4 | Labs 1-4, 6: identical corpus in all three backends, top result agreement, metadata filter agreement, hybrid search recovery |
| `scripts/module_10/benchmark_and_evaluate.py` | 4 | Labs 5-6: real latency measurement and real recall/precision/MRR/nDCG across all three backends |
| `notebooks/10_vector_search_and_local_vector_databases.ipynb` | — | **Executed end-to-end** — every cell a real measurement, no honest-skip |

**53 new tests this module** (900 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: three backends, one interface, identical results (from the executed notebook)

Query `"I forgot my password"` against the same 5-document corpus, indexed identically in all
three stores:

```
numpy:    0.463  doc_password2 | 0.183  doc_password | 0.129  doc_order_code
chroma:   0.463  doc_password2 | 0.183  doc_password | 0.129  doc_order_code
lancedb:  0.463  doc_password2 | 0.183  doc_password | 0.129  doc_order_code
```

Identical scores and ordering across brute-force NumPy, Chroma's HNSW index, and LanceDB's
vector search — at this corpus size (5 documents), ANN and exact search agree exactly, which is
the expected result: the accuracy trade-off ANN search makes only bites at larger scale, which
is exactly why §"Real proof: latency and recall" below measures rather than assumes it holds at
scale.

Metadata filtering (`category=account`) also agrees exactly across all three backends despite
being implemented three different ways (client-side dict comparison, Chroma's `where`
push-down, LanceDB's client-side JSON filter).

## Real proof: a genuine implementation bug found while building this module

While testing `add()`'s overwrite behavior (the theory doc's "Incremental updates" topic),
Chroma's plain `add()` was discovered to **silently keep the original document** on a duplicate
id instead of erroring or overwriting it — and LanceDB's plain `add()` **silently appends a
duplicate row**. Neither is documented as an obvious footgun; both were caught by testing the
overwrite-on-same-id contract Module 9's `NumpyEmbeddingIndex` already established, not assumed
to work the same way. Fixed by using each backend's real upsert primitive instead:
`collection.upsert()` for Chroma, `table.merge_insert("id").when_matched_update_all()...` for
LanceDB. The notebook's §6 demonstrates the fix working correctly across all three backends
(count unchanged after an upsert to an existing id, decremented correctly after a delete).

## Real proof: hybrid search recovers what vector search alone misses (from the executed notebook)

```
vector-only top result for query 'ACC88213': doc_password2
hybrid_search results:                       ['doc_order_code', 'doc_password2', 'doc_password']
```

`doc_order_code` ("Your order ACC88213 has shipped...") shares the exact query term with the
query but isn't the closest vector match under `FakeEmbedder`'s hashing — vector-only search
ranks a different, unrelated document first. `hybrid_search()`'s Reciprocal Rank Fusion of
vector + keyword rankings correctly surfaces `doc_order_code` as the top result instead. This is
the real, measured failure mode hybrid search exists to fix, not an assumed benefit.

## Real proof: persistence across a fresh client (from the executed notebook)

```
store1 count: 1
store2 (fresh client, same path) count: 1
```

A genuinely new `ChromaVectorStore` instance, opened against the same on-disk path after the
first instance added a document — not a mock, not the same Python object, an actual
`PersistentClient` read from disk. Same proof standard as Module 8.5's `SessionStore` restart
test.

## Real proof: latency and recall at this corpus size (from the executed notebook)

| Store | Mean latency (s) | Recall@k | Precision@k | MRR | nDCG@k |
|---|---:|---:|---:|---:|---:|
| numpy | 0.000017 | 1.00 | 0.44 | 0.83 | 0.88 |
| chroma | 0.000338 | 1.00 | 0.44 | 0.83 | 0.88 |
| lancedb | 0.001682 | 1.00 | 0.44 | 0.83 | 0.88 |

Brute-force NumPy is ~20-100x faster than Chroma or LanceDB at this scale — expected, since
both real backends pay index-management and (for LanceDB) columnar-format overhead that only
pays for itself once brute-force's O(n) scan becomes the bottleneck, which a 5-document corpus
never reaches. Recall/precision/MRR/nDCG are identical across all three because, at this scale,
ANN search returns the exact same top-k as brute force. Both numbers are the real point of this
module's evaluation discipline: **"NumPy is faster here" and "recall is unaffected here" are
measured facts about a 5-document corpus, not general claims about vector database
performance** — re-running `benchmark_and_evaluate.py` against a corpus large enough to make
Chroma/LanceDB's ANN indexing pay off (thousands+ documents) is the natural follow-up, deferred
only because this module's goal was correctness and interface parity, not a large-scale
benchmark.

## Deliberately not done in Module 10

- SQLite + vector extension (`sqlite-vec`/`sqlite-vss`) and DuckDB + Parquet + vectors are
  documented in the theory doc's options table but not implemented — three real backends
  (NumPy, Chroma, LanceDB) already prove the `VectorStore` protocol is backend-agnostic; a
  fourth and fifth would be repetition, not new signal.
- `hybrid.py`'s keyword scoring is a simple term-overlap ratio, not full BM25 — out of scope,
  a real, useful, honestly-labeled simplification in the same spirit as Module 9's `FakeEmbedder`.
- No ANN-vs-brute-force accuracy divergence demonstrated — the corpus used throughout this
  module (5 documents) is too small for Chroma/LanceDB's ANN indexes to diverge from exact
  search; a larger benchmark is the natural next step, not attempted here since this module's
  goal was interface correctness, not a performance study.
- Reranking and context packing (the next two stages of the "auth/ACL filter -> metadata
  filter -> vector/hybrid retrieval -> **rerank** -> **context pack**" pipeline) are Module 12's
  subject, not implemented here.
