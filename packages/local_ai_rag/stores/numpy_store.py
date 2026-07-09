"""NumpyVectorStore — the `VectorStore` protocol wrapped around Module 9's
`NumpyEmbeddingIndex` (theory doc "Brute-force reference store"). The
from-scratch precursor every other backend in this module is compared
against; no threading needed since the underlying work is already
in-memory NumPy, not a blocking I/O call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from local_ai_rag.embeddings.embedder import NumpyEmbeddingIndex, SearchResult


class NumpyVectorStore:
    def __init__(self) -> None:
        self._index = NumpyEmbeddingIndex()

    async def add(
        self, doc_id: str, text: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None
    ) -> None:
        self._index.add(doc_id, text, embedding, metadata=metadata)

    async def delete(self, doc_id: str) -> None:
        self._index.delete(doc_id)

    async def search(
        self, query_embedding: np.ndarray, k: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        return self._index.search(query_embedding, k=k, metadata_filter=metadata_filter)

    async def count(self) -> int:
        return len(self._index)
