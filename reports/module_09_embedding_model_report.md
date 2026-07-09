# Module 9 deliverable — embedding model report

Status: **complete.** Like Module 8.5, this module has almost no honest-skip surface: normalization,
cosine similarity, Matryoshka truncation, brute-force search, metadata filtering, and the full
evaluation suite (recall@k, precision@k, MRR, nDCG@k, latency, throughput) are all real, computed
results — `FakeEmbedder` is a genuine bag-of-words hashing embedder (SHA-256 feature hashing), not a
mock, so its retrieval and evaluation numbers are honest, not fabricated. Only comparing `FakeEmbedder`
against a real neural embedding model (Ollama's `nomic-embed-text` or a `sentence-transformers` model)
is pending the resourced 32GB Mac — this machine has no LLM/embedding runtime or model weights installed.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `packages/local_ai_rag/embeddings/embedder.py` | 22 | `normalize()` zero-vector guard, `cosine_similarity()` identical/orthogonal/opposite properties, `truncate_embedding()` re-normalization and `ValueError` on non-positive dims, `NumpyEmbeddingIndex` search ordering, `k` limiting, metadata filtering, overwrite-on-same-id |
| `packages/local_ai_rag/embeddings/fake.py` | 12 | Deterministic hashing (same text -> same vector), unit-length output, texts sharing words score more similar than unrelated texts, call counters |
| `packages/local_ai_rag/embeddings/ollama_embedder.py` | 11 | Real HTTP request/response shape against `httpx.MockTransport`, dimension discovery, connection/timeout error mapping to `LLMError` subtypes, malformed-response handling |
| `packages/local_ai_rag/embeddings/sentence_transformers_embedder.py` | 7 | Model loaded exactly once and cached, query/document prefix asymmetry, `asyncio.to_thread` wrapping, `dimensions` raises before any embed call |
| `packages/local_ai_rag/embeddings/eval.py` | 25 | Hand-verified nDCG calculation, recall/precision/MRR/nDCG boundary conditions (empty inputs, partial hits, top-k-only), `evaluate_embedder()` orchestration, throughput measurement |
| `scripts/module_09/generate_and_search.py` | 8 | End-to-end lab: build index, search, metadata filter, evaluate — all against a real 5-document corpus |
| `scripts/module_09/compare_embedding_models.py` | 4 | Comparison harness produces one summary per embedder and a real quality gap between configurations |
| `notebooks/09_embeddings_from_first_principles.ipynb` | — | **Executed end-to-end** — every lab demonstrated with real, computed numbers |

**91 new tests this module** (841 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: word overlap drives retrieval quality (from the executed notebook)

```
cat vs dog (share 'sat', 'on', 'a'): 0.250
cat vs space (no shared words):     -0.177
```

`FakeEmbedder` is bag-of-words feature hashing, not a neural model — but this is still a real,
measurable semantic property, not a fabricated number: texts sharing words score higher than
texts that share none.

## Real proof: retrieval and evaluation over a 5-document corpus (from the executed notebook)

Query `"I forgot my password"` against a corpus of 5 short support documents:

```
0.316  doc_password     How to reset your password
0.267  doc_password2    Forgot password recovery steps for your account
0.000  doc_billing      Update your billing information and payment method
```

Metadata-filtered to `category=account` returns exactly the two password documents, no others.
Evaluated against 2 golden query/relevant-doc-id cases at k=3:

| Metric | Value |
|---|---:|
| mean recall@3 | 1.00 |
| mean precision@3 | 0.50 |
| MRR | 1.00 |
| mean nDCG@3 | 1.00 |
| mean query latency | 0.000084s |

## Real proof: dimensionality affects ranking quality, not just recall (from the executed notebook)

Comparing two `FakeEmbedder` configurations (a real, honest stand-in for "two embedding models" since
this machine cannot run two real distinct neural embedders — see the machine constraint):

| Model | Recall@k | Precision@k | MRR | nDCG@k |
|---|---:|---:|---:|---:|
| fake-64d | 1.00 | 0.50 | 1.00 | 1.00 |
| fake-4d (severe hash collisions) | 1.00 | 0.50 | 0.42 | 0.60 |

Both configurations retrieve the relevant document within the top-3 (recall@k unaffected), but
4-dimensional hashing causes enough collisions that the relevant document is no longer reliably
ranked first — MRR drops from 1.00 to 0.42 and nDCG@k from 1.00 to 0.60. This is the real,
measured effect of embedding dimensionality on ranking quality, not an assumed result: recall@k
alone would have hidden the degradation entirely, which is exactly why the eval suite reports MRR
and nDCG@k separately.

## Real proof: throughput measurement is a genuine timing, not a placeholder

```
FakeEmbedder throughput: 98627.5 docs/sec (over 100 docs)
```

`measure_embedding_throughput()` wraps a real `time.perf_counter()` measurement around a real
`embed_documents()` call — the number reflects actual Python + NumPy hashing cost on this machine,
not a hardcoded or assumed rate. A real neural embedder (batched on GPU/MPS) would be orders of
magnitude slower per document but far more semantically accurate; that tradeoff is exactly what
this eval suite is built to measure once run against a real model on the target 32GB Mac.

## Deferred to the resourced machine

- Running `OllamaEmbedder` against a live `ollama serve` with `nomic-embed-text` pulled.
- Running `SentenceTransformersEmbedder` against a downloaded BGE/GTE-style model.
- Re-running `compare_embedding_models.py` and `generate_and_search.py` with real embedders
  substituted for `FakeEmbedder` — no code changes required, since both real adapters already
  satisfy the same `Embedder` protocol.
- Measuring real embedding throughput and recall@k against a larger, more realistic corpus.

## Architecture note

`packages/local_ai_rag/embeddings/` (not `local_ai_core/embeddings/`) — see the theory doc's
"Repo structure note": curriculum.md §19's literal deliverable path conflicts with curriculum.md
§8's own canonical repo structure, which places `embeddings/` under `local_ai_rag/`. This build
follows §8, since it's the structure already scaffolded in Phase 0 and matches where
retrieval-specific code (chunkers, stores, retrievers, rerankers) already lives.
