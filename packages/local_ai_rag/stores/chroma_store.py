"""ChromaVectorStore — `VectorStore` protocol backed by a real Chroma
collection (theory doc "Vector DB options": "quick local RAG" tradeoff).

Chroma's client is synchronous, so every call is wrapped in
`asyncio.to_thread` (same reasoning as Module 9's
`SentenceTransformersEmbedder`). Chroma computes distance itself rather
than accepting pre-scored results, so the collection is created with
`hnsw:space: cosine` and the returned cosine *distance* is converted back
to a cosine *similarity* score (`1 - distance`) so results are comparable
across every `VectorStore` backend, not just this one.

Chroma requires collection names of 3-512 characters from [a-zA-Z0-9._-];
`collection_name` is not validated here beyond what Chroma itself enforces.
"""

from __future__ import annotations

import asyncio
from typing import Any

import chromadb

from local_ai_rag.embeddings.embedder import SearchResult


def _build_where(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata_filter:
        return None
    if len(metadata_filter) == 1:
        (key, value), = metadata_filter.items()
        return {key: value}
    return {"$and": [{key: value} for key, value in metadata_filter.items()]}


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        *,
        path: str | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def add(
        self, doc_id: str, text: str, embedding: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Uses `upsert`, not `add` - Chroma's `add` silently keeps the
        original document on a duplicate id instead of overwriting it,
        which would violate the overwrite-on-same-id contract every other
        `VectorStore` backend honors.
        """
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[doc_id],
            embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)],
            documents=[text],
            metadatas=[metadata or None],
        )

    async def delete(self, doc_id: str) -> None:
        await asyncio.to_thread(self._collection.delete, ids=[doc_id])

    async def search(
        self, query_embedding: Any, k: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[vector],
            n_results=k,
            where=_build_where(metadata_filter),
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            SearchResult(doc_id=doc_id, score=1.0 - distance, text=text, metadata=metadata or {})
            for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)
