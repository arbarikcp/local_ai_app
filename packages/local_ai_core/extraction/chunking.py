"""Chunking and partial-extraction merging (theory doc §6, Lab 2).

A minimal, honest chunker for long documents - full retrieval-aware
chunking strategy is Module 11's job (RAG). This is scoped to what
extraction needs: keep related content together where possible, and never
silently pick one chunk's answer over another's when they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _pack_words(words: list[str], max_chars: int) -> list[str]:
    """Greedily pack words into pieces up to max_chars, splitting on
    whitespace rather than raw character count, so a hard split never cuts
    a word in half. Only a single word that itself exceeds max_chars falls
    back to a raw character split, since there's no whitespace to use.
    """
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(word) <= max_chars:
            current = word
        else:
            for i in range(0, len(word), max_chars):
                pieces.append(word[i : i + max_chars])
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars``, preferring to
    break on paragraph boundaries (``\\n\\n``) rather than mid-sentence.

    ``overlap_chars`` repeats that many trailing characters of each chunk at
    the start of the next, so a fact split across a chunk boundary has a
    chance of appearing whole in at least one chunk.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")

    paragraphs = [p for p in text.split("\n\n") if p]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= max_chars:
            current = para
        else:
            chunks.extend(_pack_words(para.split(" "), max_chars))

    if current:
        chunks.append(current)

    if overlap_chars == 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prefix = chunks[i - 1][-overlap_chars:]
        overlapped.append(prefix + chunks[i])
    return overlapped


@dataclass(frozen=True)
class MergedExtraction:
    merged: dict[str, Any]
    conflicting_fields: list[str]


def merge_partial_extractions(partials: list[dict[str, Any]]) -> MergedExtraction:
    """Merge per-chunk extraction dicts.

    First non-null value found wins per field. A field where two chunks
    give different non-null values is flagged in ``conflicting_fields``
    rather than silently overwritten - a caller (or human review, Lab 6)
    decides what to do with a genuine disagreement.
    """
    merged: dict[str, Any] = {}
    conflicts: set[str] = set()

    for partial in partials:
        for key, value in partial.items():
            if value is None:
                continue
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                conflicts.add(key)

    return MergedExtraction(merged=merged, conflicting_fields=sorted(conflicts))
