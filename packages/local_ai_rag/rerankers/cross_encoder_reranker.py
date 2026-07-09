"""CrossEncoderReranker (theory doc §11) - wraps `sentence-transformers`'
`CrossEncoder` with the same lazy-import/dependency-injection pattern
Module 9's `SentenceTransformersEmbedder` established, so tests substitute
a fake `score_fn` without the package installed or a model downloaded
(this repo's machine constraint: no model runtime or weights on this
machine). A cross-encoder scores (query, candidate) pairs jointly - more
accurate than the vector-similarity-only first pass, but too slow to run
over an entire corpus, which is why it reranks a small top-k candidate
set rather than replacing retrieval.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Callable

from local_ai_rag.embeddings.embedder import SearchResult

ScoreFn = Callable[[Any, list[tuple[str, str]]], list[float]]


def _real_load(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _real_score(model: Any, pairs: list[tuple[str, str]]) -> list[float]:
    return list(model.predict(pairs))


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        load_fn: Callable[[str], Any] = _real_load,
        score_fn: ScoreFn = _real_score,
    ) -> None:
        self.model_name = model_name
        self._load_fn = load_fn
        self._score_fn = score_fn
        self._model: Any | None = None

    async def _get_model(self) -> Any:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_fn, self.model_name)
        return self._model

    async def rerank(self, query: str, candidates: list[SearchResult], k: int | None = None) -> list[SearchResult]:
        if not candidates:
            return []
        model = await self._get_model()
        pairs = [(query, c.text) for c in candidates]
        scores = await asyncio.to_thread(self._score_fn, model, pairs)
        rescored = [replace(c, score=float(s)) for c, s in zip(candidates, scores)]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:k] if k is not None else rescored
