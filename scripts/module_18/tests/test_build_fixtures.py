import fitz
import pdfplumber

from build_fixtures import build_sample_invoice, build_scanned_receipt


class TestBuildSampleInvoice:
    def test_produces_a_real_extractable_text_layer(self, tmp_path):
        output = tmp_path / "invoice.pdf"
        build_sample_invoice(output)
        doc = fitz.open(output)
        text = doc[0].get_text()
        doc.close()
        assert "Invoice" in text

    def test_produces_a_real_extractable_table(self, tmp_path):
        output = tmp_path / "invoice.pdf"
        build_sample_invoice(output)
        with pdfplumber.open(output) as pdf:
            tables = pdf.pages[0].extract_tables()
        assert tables[0][0] == ["Item", "Qty", "Unit Price"]


class TestBuildScannedReceipt:
    def test_produces_a_page_with_no_text_layer(self, tmp_path):
        output = tmp_path / "receipt.pdf"
        build_scanned_receipt(output)
        doc = fitz.open(output)
        text = doc[0].get_text()
        doc.close()
        assert text == ""

    def test_the_page_still_contains_a_real_image(self, tmp_path):
        output = tmp_path / "receipt.pdf"
        build_scanned_receipt(output)
        doc = fitz.open(output)
        images = doc[0].get_images()
        doc.close()
        assert len(images) > 0
