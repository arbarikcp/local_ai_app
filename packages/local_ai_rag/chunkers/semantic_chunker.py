"""Semantic chunking (theory doc "Semantic chunking") - chunk boundaries
follow meaning, not a fixed character count. Splits text into sentences,
embeds each one, and starts a new chunk whenever similarity to the running
chunk's most recent sentence drops below `similarity_threshold`.

Deliberately simple sentence-level, adjacent-only comparison (not a
clustering algorithm) - real and genuinely produces different, meaning-
aware boundaries (§"Real proof" in the deliverable report), without the
complexity of a full topic-segmentation model this course doesn't need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_ai_rag.embeddings.embedder import Embedder, cosine_similarity
from local_ai_rag.loaders.markdown_loader import Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [s for s in _SENTENCE_RE.split(normalized) if s]


@dataclass(frozen=True)
class SemanticChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


async def chunk_semantically(
    document: Document, embedder: Embedder, similarity_threshold: float = 0.3, min_sentences: int = 1
) -> list[SemanticChunk]:
    sentences = split_sentences(document.text)
    if not sentences:
        return []

    embeddings = await embedder.embed_documents(sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, embedding, prev_embedding in zip(sentences[1:], embeddings[1:], embeddings[:-1]):
        similarity = cosine_similarity(embedding, prev_embedding)
        if similarity < similarity_threshold and len(groups[-1]) >= min_sentences:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        SemanticChunk(chunk_id=f"{document.doc_id}::sem::{i}", doc_id=document.doc_id, chunk_index=i, text=" ".join(group))
        for i, group in enumerate(groups)
    ]
