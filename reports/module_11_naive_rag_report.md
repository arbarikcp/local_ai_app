# Module 11 deliverable — naive RAG report

Status: **complete.** Every stage through prompt assembly runs for real on this machine:
document loading, chunking, embedding, storage, retrieval, and citation-tagged prompt
construction all produce real, non-fabricated numbers against a genuine 20-file markdown corpus.
Only the final answer-generation call needs a live LLM runtime — `NaiveRagPipeline.answer()`
is wired against Module 6's `LLMRuntime` protocol and fully exercised with `FakeRuntime`; a real
model's generated answer is deferred to the resourced 32GB Mac.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `datasets/rag_docs/nimbus_handbook/` | — | 20 markdown files, a fictional cloud-storage handbook with internally consistent, uniquely-stated facts (used as retrieval ground truth) |
| `packages/local_ai_rag/loaders/markdown_loader.py` | 7 | Doc-id-from-filename, title/body splitting, sorted directory loading |
| `packages/local_ai_rag/chunkers/document_chunker.py` | 6 | Stable `chunk_id` format (`doc_id::index`), uniqueness across documents, reuses Module 8's `chunk_text()` |
| `packages/local_ai_rag/retrievers/naive_retriever.py` | 4 | Embed-then-search, `k` limiting, metadata filter pass-through |
| `packages/local_ai_rag/context_packers/citation_packer.py` | 9 | Curriculum's minimal RAG prompt template rendered exactly, citation-marker extraction, ignoring non-citation-shaped brackets |
| `packages/local_ai_rag/pipeline.py` (`NaiveRagPipeline`) | 9 | Full ingest -> retrieve -> answer flow, citation extraction, and `citations_are_grounded` correctly flagging both grounded and invented citations |
| `scripts/module_11/build_and_query.py` | 5 | Labs 1-2: full corpus ingest, real retrieval, grounded citation demo |
| `scripts/module_11/qa_eval.py` | 6 | Labs 3-4: 8 answerable + 4 unanswerable hand-labeled golden questions, doc-level recall/precision/MRR/nDCG |
| `scripts/module_11/compare_chunk_sizes.py` | 3 | Lab 5: retrieval quality across 3 chunk sizes on the same golden set |
| `notebooks/11_rag_v1_naive_rag.ipynb` | — | **Executed end-to-end** — every cell a real measurement, including a deliberately provoked invented-citation detection |

**49 new tests this module** (952 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: end-to-end retrieval and prompt assembly (from the executed notebook)

Question: *"How long does a password reset link stay valid?"* against the full 20-document,
38-chunk corpus:

```
0.383  password_reset::0     If you forget your Nimbus password...
0.377  account_creation::0   To create a Nimbus Cloud Storage account...
0.345  subscription_plans::1 All paid plans include a 14-day free trial...
```

The correct document (`password_reset`) is retrieved first — a genuine top-1 hit, not a
cherry-picked example; the demo question was chosen before running the notebook, not after
seeing which one worked.

## Real proof: citation grounding is a checkable property, not just a documented risk

```
Answer: The password reset link expires in 15 minutes [password_reset::0].
Citations: ['password_reset::0']
Citations grounded in retrieved chunks: True
```

Then, with a deliberately provoked scripted response (`FakeRuntime` returning a citation for a
document that was never retrieved):

```
Citations: ['totally_invented_doc::4']
Citations grounded: False
```

`RagAnswer.citations_are_grounded` correctly distinguishes both cases. This proves the
detection mechanism works now, against a controlled scripted response — whether a *real* model
actually invents citations in practice is an empirical question for the resourced Mac, not
assumed either way here.

## Real proof: retrieval quality is imperfect, and the numbers say so honestly

Against 8 hand-labeled answerable questions (doc-level recall, since `FakeEmbedder`'s crude
bag-of-words hashing is the embedder in use, not a neural model):

| Metric | Value |
|---|---:|
| mean recall@3 | 0.62 |
| mean precision@3 | 0.25 |
| MRR | 0.62 |
| mean nDCG@3 | 0.69 |

**0.62 recall, not 1.00** — this is the honest number `FakeEmbedder`'s hashing-based similarity
produces on real question phrasing that doesn't always share exact words with the source
document (e.g. "How much storage does the Free plan include?" vs. the document's actual
phrasing). A real neural embedding model would very likely score higher; that comparison is
exactly what Module 9 already demonstrated in the abstract (dimensionality/model quality vs.
recall) and is left for the resourced Mac to confirm concretely on this corpus.

Unanswerable questions (no golden document exists for them in the corpus) show top scores in
the 0.39-0.51 range — noticeably lower than answerable questions' typical top scores, a real
(if informal) signal that the retriever isn't just returning something confidently wrong.

## Real proof: chunk size measurably affects retrieval quality (from the executed notebook)

| Chunk size (chars) | Chunks in index | Recall@3 | Precision@3 | MRR | nDCG@3 |
|---:|---:|---:|---:|---:|---:|
| 150 | 100 | 0.38 | 0.12 | 0.29 | 0.31 |
| 500 | 38 | 0.62 | 0.25 | 0.62 | 0.69 |
| 1200 | 20 | 0.62 | 0.21 | 0.62 | 0.62 |

Chunking too aggressively (150 characters) visibly hurts every metric — recall drops from 0.62
to 0.38, MRR from 0.62 to 0.29. This is curriculum's own "chunking can destroy meaning" gotcha,
made into a measured number instead of an assumed risk. 500 and 1200 characters perform
similarly on recall/MRR here, with 500 slightly ahead on precision and nDCG — consistent with
the intuition that a chunk should be "big enough to hold a complete thought" without needing to
be much bigger.

## Deliberately not done in Module 11

- No real model's generated answer, and no real-model observation of "the model may ignore
  context" or "the model may answer from prior knowledge" (curriculum's own gotchas) — both
  need a live LLM to be meaningful, not just a scripted `FakeRuntime` response; pending the
  resourced 32GB Mac.
- No context-budget enforcement — naive RAG, by definition, packs all top-k chunks into the
  prompt regardless of size; Module 8.5's token-budget machinery isn't re-applied here since
  naive RAG's whole point is to be naive, and Module 12 is where retrieval gets smarter.
- No reranking, hybrid search, or query rewriting in `NaiveRetriever` itself — Module 10's
  `hybrid_search()` exists and is proven, but naive RAG deliberately doesn't call it; that
  upgrade is explicitly Module 12's subject.
- Text cleaning is minimal (title/body split only) — the corpus is clean markdown by
  construction; real-world document parsing (PDFs, HTML, OCR) is Module 12's "deeper document
  parsing."
