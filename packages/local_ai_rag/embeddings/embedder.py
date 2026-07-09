"""Embedder Protocol, cosine similarity math, and NumpyEmbeddingIndex — the
from-scratch implementation (theory doc "From-scratch implementation").

texts -> embedding vectors -> normalize -> cosine similarity -> top-k retrieval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    """Query/document asymmetry (theory doc §6) is a first-class API
    distinction - embed_query() and embed_documents() are separate methods,
    not a single generic embed() a caller might misuse by embedding a query
    as if it were a document (or vice versa).
    """

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    async def embed_query(self, text: str) -> np.ndarray: ...

    @property
    def dimensions(self) -> int: ...


def normalize(v: np.ndarray) -> np.ndarray:
    """Zero-vector guard: a degenerate all-zero embedding must not raise a
    division error - returning it unchanged makes it compare as maximally
    dissimilar to everything (cosine similarity computes to 0), the correct
    degenerate behavior rather than a crash.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def truncate_embedding(v: np.ndarray, dimensions: int) -> np.ndarray:
    """Matryoshka-style truncation: keep the first `dimensions` components
    and re-normalize. Whether this preserves meaningful similarity is a
    property of the specific model's training, not something this utility
    can guarantee (theory doc "Embedding serving reality").
    """
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if dimensions >= len(v):
        return v
    return normalize(v[:dimensions])


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class NumpyEmbeddingIndex:
    """Brute-force, in-memory, NumPy-backed vector index — the from-scratch
    precursor to Module 10's real vector database comparison (LanceDB/
    Chroma/DuckDB). Supports exact-match metadata filtering (Lab 6);
    full metadata-first retrieval architecture is Module 10's subject.
    """

    def __init__(self) -> None:
        self._documents: dict[str, IndexedDocument] = {}

    def add(
        self, doc_id: str, text: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None
    ) -> None:
        self._documents[doc_id] = IndexedDocument(
            doc_id=doc_id, text=text, embedding=embedding, metadata=metadata or {}
        )

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._documents

    def search(
        self, query_embedding: np.ndarray, k: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")

        candidates = list(self._documents.values())
        if metadata_filter:
            candidates = [
                d
                for d in candidates
                if all(d.metadata.get(key) == value for key, value in metadata_filter.items())
            ]

        scored = [
            SearchResult(
                doc_id=d.doc_id,
                score=cosine_similarity(query_embedding, d.embedding),
                text=d.text,
                metadata=d.metadata,
            )
            for d in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
