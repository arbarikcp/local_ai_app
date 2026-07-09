"""LanceDBVectorStore — `VectorStore` protocol backed by a real embedded
LanceDB table (theory doc "Vector DB options": "embedded production-style
vector search" tradeoff, must learn LanceDB's own data model/indexing).

LanceDB's client is synchronous, so every call is wrapped in
`asyncio.to_thread` (same reasoning as `ChromaVectorStore`). Metadata is
stored as a JSON string column (LanceDB/Arrow have no free-form-dict column
type) and filtered client-side after a full-table vector search, since
arbitrary per-document metadata keys can't be pushed down as SQL columns
without a fixed schema per key. LanceDB's cosine `_distance` is `1 -
cosine_similarity`, converted back to a similarity score for the same
reason `ChromaVectorStore` does.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import lancedb
import pyarrow as pa

from local_ai_rag.embeddings.embedder import SearchResult

_SCHEMA_CACHE: dict[int, pa.Schema] = {}


def _schema_for(dimensions: int) -> pa.Schema:
    if dimensions not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[dimensions] = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimensions)),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
            ]
        )
    return _SCHEMA_CACHE[dimensions]


def _matches(metadata_json: str, metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    metadata = json.loads(metadata_json)
    return all(metadata.get(key) == value for key, value in metadata_filter.items())


class LanceDBVectorStore:
    def __init__(self, table_name: str, *, path: str, dimensions: int) -> None:
        self._db = lancedb.connect(path)
        self._dimensions = dimensions
        if table_name in self._db.list_tables().tables:
            self._table = self._db.open_table(table_name)
        else:
            self._table = self._db.create_table(table_name, schema=_schema_for(dimensions))

    async def add(
        self, doc_id: str, text: str, embedding: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Uses `merge_insert` (a real upsert), not `add` - plain `add`
        appends a duplicate row on a repeated id instead of overwriting it,
        which would violate the overwrite-on-same-id contract every other
        `VectorStore` backend honors.
        """
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        row = {"id": doc_id, "vector": vector, "text": text, "metadata": json.dumps(metadata or {})}
        await asyncio.to_thread(
            lambda: self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    async def delete(self, doc_id: str) -> None:
        escaped = doc_id.replace("'", "''")
        await asyncio.to_thread(self._table.delete, f"id = '{escaped}'")

    async def search(
        self, query_embedding: Any, k: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        row_count = await asyncio.to_thread(self._table.count_rows)
        if row_count == 0:
            return []
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        rows = await asyncio.to_thread(
            lambda: self._table.search(vector).metric("cosine").limit(row_count).to_list()
        )
        results = [
            SearchResult(
                doc_id=row["id"],
                score=1.0 - row["_distance"],
                text=row["text"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
            if _matches(row["metadata"], metadata_filter)
        ]
        return results[:k]

    async def count(self) -> int:
        return await asyncio.to_thread(self._table.count_rows)
