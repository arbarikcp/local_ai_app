"""Generic, reference-based retrieval metrics (theory doc §3, §6-7) -
recall@k, precision@k, MRR, nDCG@k, plus Ragas-vocabulary aliases
(`context_precision`, `context_recall`) for the same math.

Moved here from `local_ai_rag/embeddings/eval.py` (Module 9) - these four
functions have no embedding-specific coupling (they operate on plain
retrieved-id lists), so they belong in `local_ai_core/evals/` where any
task's evaluation code (RAG, extraction, later agent evaluation) can use
them, not duplicated per task package. `local_ai_rag/embeddings/eval.py`
now imports these instead of defining its own copies.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (each relevant doc contributes gain 1, not graded relevance)."""
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(top_k, start=1) if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# Ragas' own vocabulary for the same math (theory doc §6-7, Lab 4) - not new
# metrics, just the names a Ragas-style evaluation report uses.
context_precision = precision_at_k
context_recall = recall_at_k
