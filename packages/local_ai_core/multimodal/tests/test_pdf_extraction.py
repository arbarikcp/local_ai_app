from pathlib import Path

from PIL import Image

from local_ai_core.multimodal.pdf_extraction import (
    extract_layout,
    extract_tables,
    extract_text_layer,
    page_count,
    render_page_to_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SAMPLE_INVOICE = REPO_ROOT / "datasets" / "multimodal" / "sample_invoice.pdf"
SCANNED_RECEIPT = REPO_ROOT / "datasets" / "multimodal" / "scanned_receipt.pdf"


class TestPageCount:
    def test_reports_the_real_number_of_pages(self):
        assert page_count(SAMPLE_INVOICE) == 1


class TestRenderPageToImage:
    def test_returns_a_real_pil_image(self):
        image = render_page_to_image(SAMPLE_INVOICE, 0)
        assert isinstance(image, Image.Image)
        assert image.width > 0
        assert image.height > 0

    def test_higher_dpi_produces_a_larger_image(self):
        low = render_page_to_image(SAMPLE_INVOICE, 0, dpi=72)
        high = render_page_to_image(SAMPLE_INVOICE, 0, dpi=150)
        assert high.width > low.width


class TestExtractTextLayer:
    def test_extracts_real_text_from_a_digital_native_pdf(self):
        text = extract_text_layer(SAMPLE_INVOICE, 0)
        assert "Invoice" in text
        assert "INV-2026-0042" in text

    def test_a_scanned_page_has_no_extractable_text(self):
        text = extract_text_layer(SCANNED_RECEIPT, 0)
        assert text == ""


class TestExtractLayout:
    def test_returns_real_word_level_bounding_boxes(self):
        words = extract_layout(SAMPLE_INVOICE, 0)
        assert len(words) > 0
        invoice_word = next(w for w in words if w.text == "Invoice")
        assert invoice_word.x0 < invoice_word.x1
        assert invoice_word.top < invoice_word.bottom

    def test_a_scanned_page_has_no_words(self):
        words = extract_layout(SCANNED_RECEIPT, 0)
        assert words == []


class TestExtractTables:
    def test_extracts_the_real_drawn_table(self):
        tables = extract_tables(SAMPLE_INVOICE, 0)
        assert len(tables) == 1
        header, *rows = tables[0]
        assert header == ["Item", "Qty", "Unit Price"]
        assert len(rows) == 3

    def test_a_scanned_page_has_no_tables(self):
        assert extract_tables(SCANNED_RECEIPT, 0) == []
