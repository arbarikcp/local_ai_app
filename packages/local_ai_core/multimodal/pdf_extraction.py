"""Real PDF extraction (theory doc §4-6) - `PyMuPDF` for rendering pages
to images, `pdfplumber` for text layer, layout (word bounding boxes), and
table extraction. Real libraries, not LLM runtimes or model weights (same
reasoning as Module 10's `chromadb`/`lancedb`) - genuinely extracts real
content from real PDFs, no OCR involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image


def page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def render_page_to_image(pdf_path: Path, page_number: int, dpi: int = 150) -> Image.Image:
    """`page_number` is 0-indexed. Real rasterization via PyMuPDF - the
    deterministic preprocessing step before any VLM call would happen,
    if one is needed at all (theory doc's recommended pipeline).
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number]
        pixmap = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        doc.close()


def extract_text_layer(pdf_path: Path, page_number: int) -> str:
    """Real embedded text, if the PDF has any - empty string for a page
    with no text layer at all (e.g. a scanned/image-only page), not an
    error. That emptiness is exactly the signal `routing.should_use_vlm()`
    uses (theory doc §11).
    """
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_number].extract_text() or ""


@dataclass(frozen=True)
class LayoutWord:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


def extract_layout(pdf_path: Path, page_number: int) -> list[LayoutWord]:
    with pdfplumber.open(pdf_path) as pdf:
        words = pdf.pages[page_number].extract_words()
    return [LayoutWord(text=w["text"], x0=w["x0"], x1=w["x1"], top=w["top"], bottom=w["bottom"]) for w in words]


def extract_tables(pdf_path: Path, page_number: int) -> list[list[list[str | None]]]:
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_number].extract_tables()
