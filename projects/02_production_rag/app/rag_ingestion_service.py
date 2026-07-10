"""Ingestion orchestration (ARCHITECTURE.md "Data flow through
ingestion") — the first real wiring of `screen_document_for_ingestion()`
into an actual ingest-then-store path (Module 22 built the guard, but
confirmed by survey it was never called from any real ingestion pipeline
until this project). Also the first place `content_hash()`-based change
detection (Module 12's `IncrementalIndexer` algorithm) is checked against
*persistent* storage rather than an in-memory manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.security.rag_ingestion_guard import SourceTrust, screen_document_for_ingestion
from local_ai_rag.chunkers.document_chunker import chunk_document
from local_ai_rag.embeddings.embedder import Embedder
from local_ai_rag.incremental_indexer import content_hash
from local_ai_rag.loaders.markdown_loader import Document
from local_ai_rag.stores.vector_store import VectorStore

from rag_metadata_store import DocumentRecord, RagMetadataStore

DEFAULT_CHUNK_MAX_CHARS = 500


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    status: str
    chunk_count: int
    quarantine_reason: str | None = None


def _chunk_ids_for(doc_id: str, chunk_count: int) -> list[str]:
    """Chunk ids are deterministic (`document_chunker.chunk_document()`
    assigns `f"{doc_id}::{i}"` for `i` in `range(len(chunks))`), so they
    can be reconstructed from `chunk_count` alone - no separate chunk-id
    list needs to be persisted in `rag_metadata_store`.
    """
    return [f"{doc_id}::{i}" for i in range(chunk_count)]


async def ingest_document(
    document: Document,
    *,
    embedder: Embedder,
    store: VectorStore,
    metadata_store: RagMetadataStore,
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
) -> IngestionResult:
    """Every document is screened regardless of source (theory doc:
    ARCHITECTURE.md's "every upload is untrusted" rule) before it ever
    reaches the embedder or vector store. A quarantined document is
    recorded, never an exception - the request succeeded, the document
    didn't pass screening.
    """
    decision = screen_document_for_ingestion(document.text, source_trust=SourceTrust.UNTRUSTED)
    if not decision.allowed:
        metadata_store.save_document(
            DocumentRecord(
                doc_id=document.doc_id,
                source_path=document.source_path,
                title=document.title,
                status="quarantined",
                content_hash=None,
                chunk_count=0,
                quarantine_reason=decision.reason,
            )
        )
        return IngestionResult(doc_id=document.doc_id, status="quarantined", chunk_count=0, quarantine_reason=decision.reason)

    new_hash = content_hash(document.text)
    existing = metadata_store.get_document(document.doc_id)

    if existing is not None and existing.content_hash == new_hash:
        return IngestionResult(doc_id=document.doc_id, status="unchanged", chunk_count=existing.chunk_count)

    if existing is not None and existing.chunk_count > 0:
        for chunk_id in _chunk_ids_for(document.doc_id, existing.chunk_count):
            await store.delete(chunk_id)

    chunks = chunk_document(document, max_chars=chunk_max_chars)
    if chunks:
        vectors = await embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            await store.add(chunk.chunk_id, chunk.text, vector, metadata={"doc_id": chunk.doc_id, "title": document.title})

    metadata_store.save_document(
        DocumentRecord(
            doc_id=document.doc_id,
            source_path=document.source_path,
            title=document.title,
            status="ingested",
            content_hash=new_hash,
            chunk_count=len(chunks),
        )
    )
    return IngestionResult(doc_id=document.doc_id, status="ingested", chunk_count=len(chunks))


async def delete_document(doc_id: str, *, store: VectorStore, metadata_store: RagMetadataStore) -> int:
    """Returns the number of chunks removed - 0 if `doc_id` wasn't known,
    distinguishable from "known but had zero chunks" only by the caller
    also checking `metadata_store.get_document()` first, same as any
    delete-then-report-count operation in this repo.
    """
    record = metadata_store.get_document(doc_id)
    if record is None:
        return 0
    chunk_ids = _chunk_ids_for(doc_id, record.chunk_count)
    for chunk_id in chunk_ids:
        await store.delete(chunk_id)
    metadata_store.delete_document(doc_id)
    return len(chunk_ids)
