"""Parent-child chunking (theory doc "Parent-child retrieval") - "index
small, retrieve big." Parent chunks are large enough to give a generator
real context; child chunks are small enough that their embeddings match a
query precisely. `retrievers/parent_child_retriever.py` searches children
and returns parent text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from local_ai_core.extraction.chunking import chunk_text
from local_ai_rag.chunkers.document_chunker import Chunk, chunk_document
from local_ai_rag.loaders.markdown_loader import Document


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class ParentChildIndex:
    parents: dict[str, Chunk] = field(default_factory=dict)
    children: list[ChildChunk] = field(default_factory=list)

    def parent_text(self, parent_id: str) -> str:
        return self.parents[parent_id].text


def chunk_document_parent_child(
    document: Document, parent_max_chars: int, child_max_chars: int
) -> ParentChildIndex:
    if child_max_chars >= parent_max_chars:
        raise ValueError("child_max_chars must be smaller than parent_max_chars")

    parents = chunk_document(document, max_chars=parent_max_chars)
    children: list[ChildChunk] = []
    for parent in parents:
        pieces = chunk_text(parent.text, max_chars=child_max_chars)
        for i, piece in enumerate(pieces):
            children.append(
                ChildChunk(
                    chunk_id=f"{parent.chunk_id}::child::{i}",
                    parent_id=parent.chunk_id,
                    doc_id=document.doc_id,
                    text=piece,
                )
            )
    return ParentChildIndex(parents={p.chunk_id: p for p in parents}, children=children)


def chunk_documents_parent_child(
    documents: list[Document], parent_max_chars: int, child_max_chars: int
) -> ParentChildIndex:
    merged = ParentChildIndex()
    for document in documents:
        result = chunk_document_parent_child(document, parent_max_chars, child_max_chars)
        merged.parents.update(result.parents)
        merged.children.extend(result.children)
    return merged
