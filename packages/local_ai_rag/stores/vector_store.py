"""VectorStore protocol — the common interface every backend (NumPy,
Chroma, LanceDB) implements, so retrieval code (Module 11+) can swap
backends without caring which one is behind it (theory doc "Local vector
DBs", "Operational trade-offs").

All methods are async so a store backed by a real embedded database (whose
client library is synchronous) can wrap its calls in `asyncio.to_thread`
without changing the interface a caller sees - the same reasoning Module 6's
`MLXRuntime` and Module 9's `SentenceTransformersEmbedder` use for wrapping
blocking calls inside an async server.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from local_ai_rag.embeddings.embedder import SearchResult

__all__ = ["SearchResult", "VectorStore"]


class VectorStore(Protocol):
    async def add(
        self, doc_id: str, text: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None
    ) -> None: ...

    async def delete(self, doc_id: str) -> None: ...

    async def search(
        self, query_embedding: np.ndarray, k: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    async def count(self) -> int: ...
