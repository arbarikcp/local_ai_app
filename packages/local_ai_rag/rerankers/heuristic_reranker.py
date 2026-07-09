"""HeuristicReranker (theory doc §11) - a real, non-neural reranker: it
recomputes a combined vector+keyword-overlap score for every candidate and
reorders by it. Not a cross-encoder (`cross_encoder_reranker.py` is the
real-model version, honest-skip), but a genuine, measurable reordering
using Module 10's `keyword_score()` - the same reasoning `hybrid_search()`
combines vector and keyword signals, applied as a rerank stage instead of
a first-pass retrieval stage.
"""

from __future__ import annotations

from dataclasses import replace

from local_ai_rag.embeddings.embedder import SearchResult
from local_ai_rag.stores.hybrid import keyword_score


class HeuristicReranker:
    def __init__(self, keyword_weight: float = 0.5) -> None:
        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0.0 and 1.0")
        self._keyword_weight = keyword_weight

    def rerank(self, query: str, candidates: list[SearchResult], k: int | None = None) -> list[SearchResult]:
        rescored = [
            replace(
                r,
                score=(1 - self._keyword_weight) * r.score + self._keyword_weight * keyword_score(query, r.text),
            )
            for r in candidates
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:k] if k is not None else rescored
