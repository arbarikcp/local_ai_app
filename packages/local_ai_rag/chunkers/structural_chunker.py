"""Table-aware and code-aware chunking (theory doc "Table-aware chunking",
"Code-aware chunking") - one shared mechanism, not two: extract atomic
structural blocks (markdown tables, fenced code blocks) before running
Module 8's size-based `chunk_text()` on the surrounding prose, so a
size-based split never cuts through the middle of a table row or a code
block. Both structures fail the same way under naive chunking (a row or a
line gets separated from the block that gives it meaning), so both are
fixed the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_ai_core.extraction.chunking import chunk_text
from local_ai_rag.loaders.markdown_loader import Document

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")

_PLACEHOLDER = "\x00STRUCTURAL_BLOCK_{}\x00"


def _extract_tables(text: str, blocks: list[str]) -> str:
    """Consecutive `|...|` lines (a markdown table, including its header
    separator row) become one atomic block - splitting mid-table breaks
    every downstream row's meaning without its header. Appends to the same
    shared `blocks` list `_extract_code_blocks` uses, so placeholder
    indices never collide between the two block types.
    """
    lines = text.split("\n")
    output_lines: list[str] = []
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]):
            start = i
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                i += 1
            blocks.append("\n".join(lines[start:i]))
            output_lines.append(_PLACEHOLDER.format(len(blocks) - 1))
        else:
            output_lines.append(lines[i])
            i += 1
    return "\n".join(output_lines)


def _extract_code_blocks(text: str, blocks: list[str]) -> str:
    def replace(match: re.Match) -> str:
        blocks.append(match.group(0))
        return _PLACEHOLDER.format(len(blocks) - 1)

    return _CODE_FENCE_RE.sub(replace, text)


@dataclass(frozen=True)
class StructuralChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    contains_structural_block: bool


def chunk_preserving_structure(document: Document, max_chars: int, overlap_chars: int = 0) -> list[StructuralChunk]:
    blocks: list[str] = []
    text_without_code = _extract_code_blocks(document.text, blocks)
    text_without_structure = _extract_tables(text_without_code, blocks)

    pieces = chunk_text(text_without_structure, max_chars=max_chars, overlap_chars=overlap_chars)

    placeholder_re = re.compile(r"\x00STRUCTURAL_BLOCK_(\d+)\x00")

    def restore(piece: str) -> str:
        return placeholder_re.sub(lambda m: blocks[int(m.group(1))], piece)

    restored = [restore(piece) for piece in pieces]
    return [
        StructuralChunk(
            chunk_id=f"{document.doc_id}::struct::{i}",
            doc_id=document.doc_id,
            chunk_index=i,
            text=piece,
            contains_structural_block="\x00" in pieces[i],
        )
        for i, piece in enumerate(restored)
    ]
