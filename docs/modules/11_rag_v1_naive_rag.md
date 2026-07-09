# Module 11 — RAG v1: Naive RAG

> Phase: RAG · Bible reference: [curriculum.md §21](../../curriculum.md#21-module-11--rag-v1-naive-rag)

## Goal

Build RAG from scratch before using frameworks — wire Modules 9 (embeddings) and 10 (vector
stores) into a full pipeline: documents in, cited answers out.

```text
Documents
  -> chunk
  -> embed chunks
  -> store vectors
Query
  -> embed query
  -> top-k search
  -> build prompt
  -> local LLM answer
```

> **Machine note:** every stage up through prompt assembly runs for real on this machine —
> loading, chunking, embedding (`FakeEmbedder`), retrieval (`NumpyVectorStore`/
> `ChromaVectorStore`/`LanceDBVectorStore` from Module 10), and citation-aware prompt building
> are all real, non-fabricated results. **Answer generation is the one stage that needs a live
> LLM runtime** ([[project-local-ai-app-curriculum]] constraint: no Ollama/model weights on
> this machine). `NaiveRagPipeline.answer()` accepts any object satisfying Module 6's
> `LLMRuntime` protocol via dependency injection — fully exercised against `FakeRuntime` here,
> honest-skip for a real model's generated answer, pending the resourced 32GB Mac.

## Repo structure note

Same reasoning as Module 9's repo structure note: `packages/local_ai_rag/loaders/`,
`chunkers/`, `retrievers/`, `context_packers/`, and `pipeline.py` follow curriculum.md §8's
canonical structure (already scaffolded in Phase 0), not a literal re-reading of this module's
own section.

## Core topics

### 1. Document loading

`loaders/markdown_loader.py`'s `load_markdown_directory()` reads every `.md` file in a
directory into a `Document(doc_id, source_path, text)` — `doc_id` is the file's stem
(`password_reset.md` -> `password_reset`), which becomes the citation key end to end.

### 2. Text cleaning

Minimal by design: strips a leading `# Title` line into a separate `title` field (so it isn't
duplicated into every chunk's embedding) and normalizes line endings. No HTML stripping,
boilerplate removal, or deduplication — the corpus (`datasets/rag_docs/nimbus_handbook/`) is
already clean markdown; heavier cleaning belongs to whatever document-parsing step produces
markdown from PDFs/HTML in the first place (Module 12's "deeper document parsing").

### 3. Chunking

`chunkers/document_chunker.py`'s `chunk_document()` wraps Module 8's `chunk_text()`
(`packages/local_ai_core/extraction/chunking.py` — paragraph-boundary-safe, falls back to
word-boundary splitting) rather than reimplementing chunking a third time in this repo. Each
resulting piece becomes a `Chunk(chunk_id, doc_id, text, chunk_index)`, where `chunk_id =
f"{doc_id}::{chunk_index}"` — the exact string later used as a citation key (§8).

### 4. Embeddings

Module 9's `Embedder` protocol and `FakeEmbedder`, unchanged — `NaiveRagPipeline.ingest()`
calls `embed_documents()` on every chunk's text.

### 5. Retrieval

`retrievers/naive_retriever.py`'s `NaiveRetriever` is the whole "naive" in "naive RAG": embed
the query, call `VectorStore.search(query_embedding, k)`, return the results — no reranking, no
hybrid search, no query rewriting (Module 12's job). Accepts any Module 10 `VectorStore`
backend unchanged.

### 6. Prompt assembly

`context_packers/citation_packer.py`'s `build_context()` renders retrieved chunks as
`[chunk_id] text` blocks and `build_rag_prompt()` fills the curriculum's own minimal RAG
prompt template (§"Minimal RAG prompt" below) — verbatim, not paraphrased, since small local
models are sensitive to prompt wording (Module 7).

### 7. Answer generation

`NaiveRagPipeline.answer()` calls `runtime.generate(prompt)` where `runtime` satisfies Module
6's `LLMRuntime` protocol — `FakeRuntime` for tests, a real `OllamaRuntime`/`MLXRuntime` on the
resourced Mac, no pipeline code changes either way.

### 8. Basic citations

`extract_citations()` parses `[chunk_id]`-style markers out of a generated answer.
`RagAnswer.citations_are_grounded` cross-checks every cited `chunk_id` against the chunk ids
actually retrieved for that query — an invented citation (a real small-model failure mode,
curriculum's own "Citations may be invented" gotcha) is a **detectable, measurable** property,
not just a documented risk.

## Minimal RAG prompt

```text
You are a question answering assistant.
Answer only using the provided context.
If the answer is not present in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
```

`build_rag_prompt()` implements this exactly, with `{context}` filled by `build_context()`'s
citation-tagged chunk blocks.

## The corpus: `datasets/rag_docs/nimbus_handbook/`

20 markdown files, a fictional cloud-storage product's support handbook (account management,
billing, file sharing, sync clients, API docs, security). Real, internally consistent facts
(exact numbers, expiry windows, plan names) so retrieval quality and unanswerable-question
detection have a genuine ground truth, not vibes — e.g. "password reset links expire in 15
minutes" is stated in exactly one document and nowhere else, so a retrieval miss on that
question is unambiguous.

## Gotchas

- **Chunking can destroy meaning** — `chunk_text()`'s paragraph-boundary preference reduces
  but does not eliminate this; §"Real proof" below measures how chunk size trades off against
  retrieval quality rather than asserting a "right" chunk size.
- **Top-k can return irrelevant chunks** — visible directly in `NaiveRetriever` output; no
  reranking (Module 12) means a mediocre chunk can occupy a top-k slot a better one deserved.
- **The model may ignore context** / **answer from prior knowledge** — both need a real model
  to observe; deferred to the resourced Mac, tracked explicitly rather than assumed away.
- **Citations may be invented** — `RagAnswer.citations_are_grounded` makes this checkable
  against `FakeRuntime`'s scripted responses now, and against a real model's output later,
  without changing the checking code.
- **Long context can reduce answer quality** — Module 6.5's/8.5's context-budget machinery
  already exists for this; `NaiveRagPipeline` doesn't re-solve it, just doesn't yet apply it
  (naive RAG, by definition, packs all top-k chunks in).

## Hands-on labs

1. **Build naive RAG over 20 markdown files** — `NaiveRagPipeline` over
   `datasets/rag_docs/nimbus_handbook/`, `scripts/module_11/build_and_query.py`.
2. **Add citations using chunk IDs** — `citation_packer.py` + `extract_citations()`.
3. **Test with answerable and unanswerable questions** — `scripts/module_11/qa_eval.py`,
   a golden set of both kinds of questions.
4. **Measure retrieval quality manually** — same script, recall@k against hand-labeled
   relevant chunk ids (Module 9's `eval.py` metric functions, reused not reimplemented).
5. **Compare 3 chunk sizes** — `scripts/module_11/compare_chunk_sizes.py`.

## Deliverable

```text
datasets/rag_docs/nimbus_handbook/         # 20 markdown files
packages/local_ai_rag/
  loaders/markdown_loader.py
  chunkers/document_chunker.py
  retrievers/naive_retriever.py
  context_packers/citation_packer.py
  pipeline.py
  tests/ (per subpackage)
scripts/module_11/
  build_and_query.py
  qa_eval.py
  compare_chunk_sizes.py
reports/module_11_naive_rag_report.md
```
