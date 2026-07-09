from pathlib import Path

from local_ai_rag.loaders.pdf_loader import load_pdf_directory, load_pdf_document

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MULTIMODAL_DIR = REPO_ROOT / "datasets" / "multimodal"
SAMPLE_INVOICE = MULTIMODAL_DIR / "sample_invoice.pdf"
SCANNED_RECEIPT = MULTIMODAL_DIR / "scanned_receipt.pdf"


class TestLoadPdfDocument:
    def test_one_document_per_page(self):
        documents = load_pdf_document(SAMPLE_INVOICE)
        assert len(documents) == 1

    def test_doc_id_encodes_the_page_number(self):
        documents = load_pdf_document(SAMPLE_INVOICE)
        assert documents[0].doc_id == "sample_invoice::page1"

    def test_text_is_the_real_extracted_content(self):
        documents = load_pdf_document(SAMPLE_INVOICE)
        assert "Invoice" in documents[0].text

    def test_a_scanned_pdf_produces_a_document_with_empty_text(self):
        documents = load_pdf_document(SCANNED_RECEIPT)
        assert documents[0].text == ""

    def test_title_is_the_pdf_stem(self):
        documents = load_pdf_document(SAMPLE_INVOICE)
        assert documents[0].title == "sample_invoice"


class TestLoadPdfDirectory:
    def test_loads_every_pdf_in_the_directory(self):
        documents = load_pdf_directory(MULTIMODAL_DIR)
        doc_ids = {d.doc_id for d in documents}
        assert "sample_invoice::page1" in doc_ids
        assert "scanned_receipt::page1" in doc_ids

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert load_pdf_directory(tmp_path) == []
