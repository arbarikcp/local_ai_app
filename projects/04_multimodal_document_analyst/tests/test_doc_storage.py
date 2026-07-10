import pytest

from doc_storage import DocStorage, DocumentRecord, PageAnalysisRecord


@pytest.fixture
def storage(tmp_path):
    with DocStorage(tmp_path / "multimodal.db") as store:
        yield store


class TestDocumentPersistence:
    def test_save_and_get_document_round_trips(self, storage):
        storage.save_document(
            DocumentRecord(
                doc_id="multi_page_form",
                source_path="datasets/multimodal/project_04/multi_page_form.pdf",
                page_count=3,
                status="ingested",
            )
        )
        record = storage.get_document("multi_page_form")
        assert record is not None
        assert record.page_count == 3
        assert record.status == "ingested"
        assert record.ingested_at is not None

    def test_get_document_returns_none_when_missing(self, storage):
        assert storage.get_document("nonexistent") is None

    def test_save_document_upserts_on_conflict(self, storage):
        storage.save_document(
            DocumentRecord(doc_id="doc1", source_path="a.pdf", page_count=1, status="ingested")
        )
        storage.save_document(
            DocumentRecord(doc_id="doc1", source_path="a.pdf", page_count=3, status="ingested")
        )
        record = storage.get_document("doc1")
        assert record.page_count == 3

    def test_save_document_rejects_invalid_status(self, storage):
        with pytest.raises(ValueError):
            storage.save_document(
                DocumentRecord(doc_id="doc1", source_path="a.pdf", page_count=1, status="bogus")
            )


class TestPageAnalysisPersistence:
    def test_save_and_get_page_analysis_round_trips(self, storage):
        storage.save_page_analysis(
            PageAnalysisRecord(
                page_id="multi_page_form::page1",
                doc_id="multi_page_form",
                page_number=1,
                route="text_llm",
                route_reason="text layer has 123 chars (>= 40 threshold) - a VLM is unnecessary",
                extracted_text="Nimbus Cloud Storage - Account Closure Request",
                extracted_fields={"applicant_name": "Jordan Rivera"},
                confidence="high",
                needs_review=False,
            )
        )
        record = storage.get_page_analysis("multi_page_form::page1")
        assert record is not None
        assert record.route == "text_llm"
        assert record.extracted_fields == {"applicant_name": "Jordan Rivera"}
        assert record.needs_review is False

    def test_page_analysis_with_no_extracted_fields_round_trips_none(self, storage):
        storage.save_page_analysis(
            PageAnalysisRecord(
                page_id="multi_page_form::page3",
                doc_id="multi_page_form",
                page_number=3,
                route="vlm",
                route_reason="text layer has only 0 chars (< 40 threshold) - likely scanned/image-only",
                extracted_text="This is a fake VLM description.",
                extracted_fields=None,
                confidence=None,
                needs_review=True,
                quarantine_reason=None,
            )
        )
        record = storage.get_page_analysis("multi_page_form::page3")
        assert record.extracted_fields is None
        assert record.needs_review is True

    def test_list_page_analyses_orders_by_page_number(self, storage):
        for page_number in (2, 1, 3):
            storage.save_page_analysis(
                PageAnalysisRecord(
                    page_id=f"multi_page_form::page{page_number}",
                    doc_id="multi_page_form",
                    page_number=page_number,
                    route="text_llm",
                    route_reason="reason",
                    extracted_text="text",
                )
            )
        records = storage.list_page_analyses("multi_page_form")
        assert [r.page_number for r in records] == [1, 2, 3]

    def test_save_page_analysis_upserts_on_conflict(self, storage):
        storage.save_page_analysis(
            PageAnalysisRecord(
                page_id="doc::page1",
                doc_id="doc",
                page_number=1,
                route="text_llm",
                route_reason="first",
                extracted_text="v1",
            )
        )
        storage.save_page_analysis(
            PageAnalysisRecord(
                page_id="doc::page1",
                doc_id="doc",
                page_number=1,
                route="text_llm",
                route_reason="second",
                extracted_text="v2",
            )
        )
        record = storage.get_page_analysis("doc::page1")
        assert record.extracted_text == "v2"
        assert record.route_reason == "second"
