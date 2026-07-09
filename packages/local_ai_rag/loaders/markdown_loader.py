"""Markdown document loading (theory doc "Document loading", "Text
cleaning"). Deliberately minimal - the corpus this course ships
(`datasets/rag_docs/nimbus_handbook/`) is already clean markdown; heavier
cleaning (HTML stripping, boilerplate removal, PDF/HTML parsing) belongs to
whatever upstream step produces markdown in the first place, which is
Module 12's "deeper document parsing" territory, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    title: str
    text: str


def _split_title(raw_text: str) -> tuple[str, str]:
    """A leading `# Title` line becomes a separate title field rather than
    being embedded in every chunk's text - repeating the title in every
    chunk would dilute the chunk's own content in its embedding.
    """
    lines = raw_text.splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        return title, body
    return "", raw_text.strip()


def load_markdown_file(path: Path) -> Document:
    raw_text = path.read_text(encoding="utf-8")
    title, body = _split_title(raw_text)
    return Document(doc_id=path.stem, source_path=str(path), title=title, text=body)


def load_markdown_directory(directory: Path) -> list[Document]:
    """Sorted by doc_id for deterministic ordering - callers (and tests)
    shouldn't depend on filesystem iteration order.
    """
    paths = sorted(Path(directory).glob("*.md"))
    return [load_markdown_file(path) for path in paths]
