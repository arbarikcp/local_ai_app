"""NaiveRetriever (theory doc "Retrieval") - embed the query, search the
vector store, return the top-k results. No reranking, no hybrid search, no
query rewriting - that's the entire meaning of "naive" here, and Module
12's job to improve on.
"""

from __future__ import annotations

from typing import Any

from local_ai_rag.embeddings.embedder import Embedder, SearchResult
from local_ai_rag.stores.vector_store import VectorStore


class NaiveRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(
        self, query: str, k: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        query_embedding = await self._embedder.embed_query(query)
        return await self._store.search(query_embedding, k=k, metadata_filter=metadata_filter)
