"""Retrieval-scoped document chunking (theory doc "Chunking"). Wraps
Module 8's `chunk_text()` (`local_ai_core.extraction.chunking`) rather than
reimplementing paragraph/word-boundary-safe splitting a third time in this
repo, and adds the doc-id/chunk-id bookkeeping retrieval needs that
extraction's chunker didn't: a stable `chunk_id` used as the citation key
end to end (theory doc "Basic citations").
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.extraction.chunking import chunk_text
from local_ai_rag.loaders.markdown_loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


def chunk_document(document: Document, max_chars: int, overlap_chars: int = 0) -> list[Chunk]:
    pieces = chunk_text(document.text, max_chars=max_chars, overlap_chars=overlap_chars)
    return [
        Chunk(chunk_id=f"{document.doc_id}::{i}", doc_id=document.doc_id, chunk_index=i, text=piece)
        for i, piece in enumerate(pieces)
    ]


def chunk_documents(documents: list[Document], max_chars: int, overlap_chars: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars))
    return chunks
