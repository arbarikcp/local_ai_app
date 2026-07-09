"""ParentChildRetriever (theory doc "Parent-child retrieval", Lab 1) -
searches child chunk embeddings (precise matches) but returns deduplicated
parent chunk text (real generation context) - "index small, retrieve big."
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_rag.chunkers.parent_child_chunker import ParentChildIndex
from local_ai_rag.embeddings.embedder import Embedder
from local_ai_rag.stores.vector_store import VectorStore


@dataclass(frozen=True)
class ParentResult:
    parent_id: str
    text: str
    best_child_score: float


class ParentChildRetriever:
    """`store` must be indexed on children (see `scripts/module_12/parent_child_demo.py`
    for the indexing side); `index` supplies the parent_id -> parent text lookup.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, index: ParentChildIndex) -> None:
        self._embedder = embedder
        self._store = store
        self._index = index

    async def retrieve(self, query: str, k: int = 5, fetch_k: int = 20) -> list[ParentResult]:
        query_embedding = await self._embedder.embed_query(query)
        child_results = await self._store.search(query_embedding, k=fetch_k)

        best_score_by_parent: dict[str, float] = {}
        for result in child_results:
            parent_id = result.metadata["parent_id"]
            if parent_id not in best_score_by_parent or result.score > best_score_by_parent[parent_id]:
                best_score_by_parent[parent_id] = result.score

        ranked_parent_ids = sorted(best_score_by_parent, key=lambda pid: best_score_by_parent[pid], reverse=True)
        return [
            ParentResult(parent_id=pid, text=self._index.parent_text(pid), best_child_score=best_score_by_parent[pid])
            for pid in ranked_parent_ids[:k]
        ]
