# Outro — Project 2: Production Local RAG Service

## What this achieved

A real, running FastAPI RAG service over private documents — ingestion with injection screening,
persistent metadata and real vector storage, question answering with citations that are actually
verified (not just extracted), and an evaluation command that produces honest, checkable numbers
including a genuinely limited retrieval recall from the honest-skip embedder. It's also this
course's clearest demonstration of the "compose, don't rebuild" discipline: five modules'
(9-13) worth of real, already-tested RAG machinery — chunking strategies, rerankers, hybrid
search, query rewriting, parent-child retrieval, citation packing — came together with almost no
new pipeline code, only the persistence and API layer a real service actually needs.

## What's still open (honest-skip, not forgotten)

- **Real embedding and generation quality.** Every metric in REPORT.md is mechanically real;
  none of them say anything about how well a real embedding model retrieves the right document or
  a real LLM writes a faithful answer. That number only exists once this runs on the resourced
  32GB Mac with `SentenceTransformersEmbedder`/`OllamaEmbedder` and a real runtime injected via
  `build_rag_context(..., embedder=..., runtime=...)` — no other code changes needed, by design.
- **The 50% recall@k is a real, informative number worth revisiting.** It's not a bug, but it's
  also not a result to be satisfied with — it's exactly the kind of number that should improve
  substantially once a real embedder replaces `FakeEmbedder`'s bag-of-words hashing, and the eval
  harness is already built to measure that improvement the moment it happens.
- **Alternate chunking strategies aren't exposed through the API.** `parent_child_chunker.py`,
  `structural_chunker.py`, and `semantic_chunker.py` are all real and tested (Module 12) but
  `/documents` only ever uses fixed-size chunking. A `chunking_strategy` request field routing to
  the right chunker is a small, well-scoped extension.
- **No pre-retrieval unanswerable-question classifier.** Abstention currently relies entirely on
  the prompt template's own instruction plus post-hoc evaluation (`refusal_check()`). A real
  classifier would let the service short-circuit before running retrieval at all for a question
  the corpus obviously can't answer.

## What to explore next

- **A real embedder comparison, once on the resourced Mac**: run the exact same evaluation
  harness against `FakeEmbedder`, `SentenceTransformersEmbedder`, and `OllamaEmbedder` on the
  identical golden set and corpus — the recall@k delta between a real neural embedder and this
  project's honest-skip default would be a genuinely interesting, load-bearing number for anyone
  deciding whether local embeddings are "good enough."
- **Hybrid search and cross-encoder reranking, activated for real.** `stores/hybrid.py`'s
  `hybrid_search()` (RRF-fused vector+keyword) and `rerankers/cross_encoder_reranker.py` both
  exist and are real (the cross-encoder is honest-skip pending `sentence-transformers`); this
  project's `ProductionRagPipeline` wiring currently uses vector-only retrieval and the heuristic
  reranker by default. Given the observed recall gap, hybrid search in particular is a strong
  candidate to try first — keyword matching doesn't depend on embedding quality at all.
- **Parent-child retrieval activated for real**, given how short and topically dense the Nimbus
  handbook's individual pages are — "index small, retrieve big" (Module 12's own framing) could
  directly address the precision@k number, which is low partly because whole short documents
  compete as single undifferentiated chunks.
- **Query rewriting/HyDE, measured against the golden set.** `retrievers/query_expansion.py`'s
  `rewrite_query()`/`hyde_retrieve()` are real and available via `answer_question(...,
  rewrite=True)` but this project's own eval run doesn't exercise that path — a natural follow-up
  experiment once a real runtime makes rewriting meaningfully different from the original query.
- **A real LLM-as-judge faithfulness pass**, using Module 13's `LocalJudge` (calibrated via
  `judge_calibration.py`) as a second, model-backed faithfulness signal alongside the word-overlap
  heuristic already wired in — the two disagreeing on a specific answer would be a genuinely
  useful signal about where the heuristic's known weaknesses actually bite.
