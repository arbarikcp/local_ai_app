import pytest
from pydantic import ValidationError

from rag_api_schemas import (
    CitationResponse,
    DeleteResponse,
    DocumentIngestRequest,
    QueryResponse,
    TraceResponse,
)


class TestDocumentIngestRequest:
    def test_accepts_a_valid_source_type(self):
        request = DocumentIngestRequest(source_type="markdown", source_path="/docs/a.md")
        assert request.source_type == "markdown"

    def test_rejects_an_unknown_source_type(self):
        with pytest.raises(ValidationError):
            DocumentIngestRequest(source_type="pptx")


class TestQueryResponse:
    def test_builds_a_full_response_matching_curriculums_shape(self):
        response = QueryResponse(
            answer="The reset link expires after 24 hours.",
            citations=[
                CitationResponse(
                    document_id="doc-1", chunk_id="doc-1::0", score=0.82, text_preview="...", verified=True
                )
            ],
            trace=TraceResponse(retrieved_chunks=12, reranked_chunks=5, context_tokens=3100, model="fake-model"),
        )
        assert response.citations[0].document_id == "doc-1"
        assert response.trace.retrieved_chunks == 12


class TestDeleteResponseValidation:
    def test_negative_chunks_removed_is_rejected(self):
        with pytest.raises(ValidationError):
            DeleteResponse(deleted=True, chunks_removed=-1)
