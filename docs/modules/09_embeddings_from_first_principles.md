# Module 9 — Embeddings from First Principles

> Phase: Application primitives · Bible reference: [curriculum.md §19](../../curriculum.md#19-module-9--embeddings-from-first-principles)

## Goal

Understand embeddings deeply enough to design RAG systems — this module is the foundation
Modules 10-13 (vector search, naive RAG, production RAG, RAG evaluation) build on directly.

> **Machine note:** this repo is built on a Mac that must never run a model runtime or
> download model weights ([[project-local-ai-app-curriculum]] constraint; target execution
> hardware confirmed as a separate 32GB Mac). Real embedding models (`sentence-transformers`,
> Ollama's embedding endpoint) are wrapped by adapters that use the same lazy-import /
> dependency-injection pattern Module 6's `MLXRuntime` and `OllamaRuntime` established —
> fully unit-tested via `FakeEmbedder` and `httpx.MockTransport`, never actually run here.

## Repo structure note

The curriculum's literal deliverable path for this module is
`packages/local_ai_core/embeddings/`; this build uses `packages/local_ai_rag/embeddings/`
instead, matching curriculum.md §8's own canonical repo structure (which places
`embeddings/`, `chunkers/`, `stores/`, `retrievers/`, `rerankers/`, and `context_packers/`
under `local_ai_rag`, not `local_ai_core`) and the directory this repo already scaffolded in
Phase 0. `local_ai_core` stays scoped to runtime/prompt/extraction/conversation/gateway
infrastructure that isn't RAG-specific.

## 1. What embeddings represent

An embedding is a fixed-length numeric vector such that semantically similar text maps to
nearby vectors under some distance measure. It is not a compressed copy of the text — most
of the original text's exact wording is unrecoverable from the vector alone. What survives
is *meaning*, approximately, as the model's training defined it — which is why embedding
model choice (§8) matters as much as any other model choice in this course.

## 2. Embedding dimensionality

Higher dimensionality can capture more distinctions but costs more storage, memory bandwidth,
and comparison time — directly relevant on the RAM-constrained Macs this course targets.
Common local embedding models range from ~384 to ~1024+ dimensions. §"Embedding serving
reality" below covers a lever for trading dimensionality against quality.

## 3-4. Cosine similarity and dot product

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))
```

Cosine similarity is the dot product of two **normalized** vectors — it measures the angle
between them, ignoring magnitude. For embeddings whose magnitude doesn't carry meaning (true
for most sentence embedding models), cosine similarity and normalized dot product are the
same measure; `embedder.py` implements both explicitly so the relationship is visible in
code, not just asserted in prose.

## 5. Normalization

```python
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
```

The zero-vector guard matters: a degenerate all-zero embedding (which a broken adapter or an
empty-string input could produce) must not raise a division error — it should compare as
maximally dissimilar to everything, which returning the zero vector unchanged achieves (its
cosine similarity to anything, including itself, computes to 0).

## 6. Query/document asymmetry

Many embedding models (E5-style, and others) are trained with different instructions or
prefixes for queries versus documents being indexed — encoding "the thing being searched
for" differently from "the thing being searched over." `embedder.py`'s `Embedder` protocol
makes this a first-class API distinction (`embed_query()` vs. `embed_documents()`), not a
single generic `embed()` a caller might misuse by embedding a query as if it were a document.

## 7. Chunking and embeddings

An embedding represents a whole input as one vector — a long document embedded as a single
vector loses fine-grained retrievability (a fact in paragraph 40 gets diluted by paragraphs
1-39). This is why RAG systems chunk documents before embedding them. Module 8's
`chunking.py` (paragraph/word-boundary-safe splitting) is the same kind of tool this need
calls for; Module 11 (naive RAG) is where chunking-for-retrieval gets its own treatment.

## 8. Embedding model choice

Same discipline as Module 3's model selection: benchmark, don't assume. §"Embedding serving
reality" below is this module's specific contribution to that process.

## 9. Multilingual embeddings

Not every embedding model handles non-English text well, and multilingual-capable models
often trade some English-only quality for that breadth. Treated here as a model-catalog
selection dimension (Module 3's `models/MODEL_CATALOG.md` pattern extends to embedding
models), not implemented as separate code in this module.

## 10. Embedding drift

If you re-embed a corpus with a different model version (or even the same model at a
different quantization/precision), old and new vectors are not guaranteed comparable —
mixing them in one index silently corrupts retrieval quality. This is precisely why Module
6.5's `EmbeddingCache` keys on `(text, embedding_model, normalization_version)`: an
embedding-model change is a cache-key change, not a silent overwrite.

## 11. Evaluation

Covered in depth below (§"Embedding evaluation").

## Embedding serving reality

**Do not assume every strong embedding model should be served through the same runtime as
the generator.** Many strong embedders (BGE/GTE/ModernBERT-style models especially) are best
run with `sentence-transformers` or Transformers directly. Ollama-style embedding endpoints
are convenient, but convenience should be benchmarked against quality, throughput, memory
residency, and vector dimensionality — not assumed. `ollama_embedder.py` and
`sentence_transformers_embedder.py` both implement the same `Embedder` protocol specifically
so this comparison (Lab 5) is an apples-to-apples swap of one adapter for another, not a
rewrite.

**Matryoshka-style embeddings**: some models allow truncating embedding dimensions to trade
retrieval quality for storage, memory bandwidth, and latency — `embedder.py`'s
`truncate_embedding()` implements the (re-normalize after truncating) mechanics generically;
whether a given model's vectors remain meaningful after truncation is a property of that
specific model's training, not something this utility can guarantee.

## From-scratch implementation

```text
texts -> embedding vectors -> normalize -> cosine similarity -> top-k retrieval
```

`embedder.py`'s `NumpyEmbeddingIndex` implements exactly this — brute-force, in-memory,
NumPy-backed — as the "from scratch" precursor to Module 10's real vector database
comparison (LanceDB/Chroma/DuckDB). It also supports the metadata filtering Lab 6 asks for,
kept intentionally minimal (exact-match filters) since full metadata-first retrieval
architecture is Module 10's subject.

## Embedding evaluation

Golden test-set shape (curriculum's own example, `eval.py`'s `EmbeddingEvalCase`):

```json
{
  "query": "How do I reset my password?",
  "positive_doc_ids": ["doc_12", "doc_18"],
  "negative_doc_ids": ["doc_03", "doc_44"]
}
```

Metrics implemented in `eval.py`: recall@k, precision@k, MRR (mean reciprocal rank), nDCG@k
(binary relevance), latency, and embedding throughput — the exact list the curriculum names.

## Hands-on labs

1. **Generate embeddings locally** — `scripts/module_09/generate_and_search.py`, against
   `FakeEmbedder` for the infrastructure proof; real embedder honest-skip.
2. **Store them in NumPy** — `NumpyEmbeddingIndex`.
3. **Search using brute force** — same class's `search()`.
4. **Evaluate recall@k** — `eval.py` + the same lab script.
5. **Compare two embedding models** — `scripts/module_09/compare_embedding_models.py`.
6. **Add metadata filtering** — `NumpyEmbeddingIndex.search(..., metadata_filter=...)`.

## Deliverable

```text
packages/local_ai_rag/embeddings/
  embedder.py
  fake.py
  ollama_embedder.py
  sentence_transformers_embedder.py
  eval.py
  tests/
scripts/module_09/
  generate_and_search.py
  compare_embedding_models.py
reports/module_09_embedding_model_report.md
```
