"""Incremental indexing (theory doc §16, Lab 6) - diff a document set
against a stored content-hash manifest so re-ingesting an unchanged corpus
costs nothing (no re-embedding), a changed document is fully re-chunked
and re-embedded (its old chunks deleted first, not left stale alongside
the new ones), and a document no longer present has its chunks deleted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from local_ai_rag.chunkers.document_chunker import chunk_document
from local_ai_rag.embeddings.embedder import Embedder
from local_ai_rag.loaders.markdown_loader import Document
from local_ai_rag.stores.vector_store import VectorStore


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IndexDiff:
    added: list[str]
    updated: list[str]
    deleted: list[str]
    unchanged: list[str]


class IncrementalIndexer:
    def __init__(self, embedder: Embedder, store: VectorStore, *, chunk_max_chars: int = 500) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_max_chars = chunk_max_chars
        self._manifest: dict[str, str] = {}
        self._chunk_ids_by_doc: dict[str, list[str]] = {}

    def diff(self, documents: list[Document]) -> IndexDiff:
        current_ids = {d.doc_id for d in documents}
        added, updated, unchanged = [], [], []
        for document in documents:
            new_hash = content_hash(document.text)
            if document.doc_id not in self._manifest:
                added.append(document.doc_id)
            elif self._manifest[document.doc_id] != new_hash:
                updated.append(document.doc_id)
            else:
                unchanged.append(document.doc_id)
        deleted = [doc_id for doc_id in self._manifest if doc_id not in current_ids]
        return IndexDiff(added=added, updated=updated, deleted=deleted, unchanged=unchanged)

    async def sync(self, documents: list[Document]) -> IndexDiff:
        diff = self.diff(documents)
        documents_by_id = {d.doc_id: d for d in documents}

        for doc_id in [*diff.deleted, *diff.updated]:
            for chunk_id in self._chunk_ids_by_doc.get(doc_id, []):
                await self._store.delete(chunk_id)
            self._chunk_ids_by_doc.pop(doc_id, None)
            self._manifest.pop(doc_id, None)

        for doc_id in [*diff.added, *diff.updated]:
            document = documents_by_id[doc_id]
            chunks = chunk_document(document, max_chars=self._chunk_max_chars)
            if chunks:
                vectors = await self._embedder.embed_documents([c.text for c in chunks])
                for chunk, vector in zip(chunks, vectors):
                    await self._store.add(chunk.chunk_id, chunk.text, vector, metadata={"doc_id": chunk.doc_id})
            self._chunk_ids_by_doc[doc_id] = [c.chunk_id for c in chunks]
            self._manifest[doc_id] = content_hash(document.text)

        return diff
