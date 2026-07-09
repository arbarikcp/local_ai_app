"""Real PDF document loading (theory doc §9, "Multimodal RAG") - one
`Document` per page, using Module 18's real `pdfplumber`-based text
extraction. Reuses Module 11's exact `Document` shape unchanged so the
existing chunker/pipeline/citation machinery needs no changes at all -
this is a new *loader*, not a new RAG architecture. Page numbers are
encoded in `doc_id` (`f"{pdf_stem}::page{n}"`, 1-indexed) rather than a
new metadata field - the same "::"-separated citation-key convention
`document_chunker.py`'s `chunk_id` already uses.
"""

from __future__ import annotations

from pathlib import Path

from local_ai_core.multimodal.pdf_extraction import extract_text_layer, page_count
from local_ai_rag.loaders.markdown_loader import Document


def load_pdf_document(pdf_path: Path) -> list[Document]:
    stem = Path(pdf_path).stem
    n_pages = page_count(pdf_path)
    documents = []
    for page_number in range(n_pages):
        text = extract_text_layer(pdf_path, page_number)
        documents.append(
            Document(doc_id=f"{stem}::page{page_number + 1}", source_path=str(pdf_path), title=stem, text=text)
        )
    return documents


def load_pdf_directory(directory: Path) -> list[Document]:
    """Sorted by filename for deterministic ordering, same convention as
    `markdown_loader.load_markdown_directory()`.
    """
    documents: list[Document] = []
    for path in sorted(Path(directory).glob("*.pdf")):
        documents.extend(load_pdf_document(path))
    return documents
