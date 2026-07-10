import pytest
from pydantic import ValidationError

from doc_schemas import (
    CitationResponse,
    DocumentFieldExtraction,
    ExtractRequest,
    IngestDocumentRequest,
    PageIngestResult,
    QueryRequest,
    QueryResponse,
)


class TestDocumentFieldExtraction:
    def test_accepts_a_fully_populated_form(self):
        extraction = DocumentFieldExtraction.model_validate(
            {
                "document_type": "account_closure_request",
                "applicant_name": "Jordan Rivera",
                "key_date": "2026-06-15",
                "key_amount": 42.50,
                "notes": "Refund owed on closure.",
                "confidence": "high",
                "evidence": {"applicant_name": "Applicant Name: Jordan Rivera"},
            }
        )
        assert extraction.applicant_name == "Jordan Rivera"
        assert extraction.key_amount == 42.50

    def test_every_field_but_confidence_is_optional(self):
        extraction = DocumentFieldExtraction.model_validate({"confidence": "low"})
        assert extraction.document_type is None
        assert extraction.evidence == {}

    def test_confidence_is_required(self):
        with pytest.raises(ValidationError):
            DocumentFieldExtraction.model_validate({})

    def test_confidence_is_constrained_to_known_levels(self):
        with pytest.raises(ValidationError):
            DocumentFieldExtraction.model_validate({"confidence": "extremely-sure"})


class TestApiSchemas:
    def test_ingest_document_request_requires_source_path(self):
        with pytest.raises(ValidationError):
            IngestDocumentRequest.model_validate({})
        request = IngestDocumentRequest.model_validate({"source_path": "datasets/multimodal/project_04/multi_page_form.pdf"})
        assert request.source_path.endswith(".pdf")

    def test_page_ingest_result_shape(self):
        result = PageIngestResult.model_validate(
            {"page_id": "multi_page_form::page1", "route": "text_llm", "status": "ingested"}
        )
        assert result.page_id == "multi_page_form::page1"

    def test_extract_request_page_number_is_optional(self):
        request = ExtractRequest.model_validate({})
        assert request.page_number is None

    def test_query_request_requires_question(self):
        with pytest.raises(ValidationError):
            QueryRequest.model_validate({})

    def test_query_response_composes_citations(self):
        response = QueryResponse.model_validate(
            {
                "answer": "The refund amount owed is $42.50.",
                "citations": [
                    {"page_id": "multi_page_form::page2", "page_number": 2, "verified": True}
                ],
            }
        )
        assert isinstance(response.citations[0], CitationResponse)
        assert response.citations[0].verified is True
