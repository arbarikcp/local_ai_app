"""Plain-text (.txt) loader (ARCHITECTURE.md, curriculum's functional
requirement 1 - "ingest markdown, text, and PDF-derived text"). Only
markdown and PDF loaders exist in `local_ai_rag/loaders/` (confirmed by
survey); this fills the one missing input format, reusing the exact same
`Document` shape unchanged so downstream chunking/embedding/storage code
never needs to know which loader produced a given document.
"""

from __future__ import annotations

from pathlib import Path

from local_ai_rag.loaders.markdown_loader import Document


def load_text_file(path: str | Path) -> Document:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return Document(doc_id=path.stem, source_path=str(path), title=path.stem, text=text)


def load_text_string(doc_id: str, text: str, *, title: str | None = None) -> Document:
    """For inline text submitted directly in a request body, with no file
    on disk at all - `POST /documents`'s `source_type: "text"` path.
    `source_path` is `""` (empty string, not `None`) since `Document.source_path`
    is a required `str`, not `str | None` - reused unchanged from Module 9.
    """
    return Document(doc_id=doc_id, source_path="", title=title or doc_id, text=text)


def load_text_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    return [load_text_file(path) for path in sorted(directory.glob("*.txt"))]
