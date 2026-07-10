import json

import pytest
from local_ai_core.extraction.pipeline import ExtractionPipeline
from local_ai_core.runtimes.fake import FakeRuntime

from doc_extraction import extract_page_fields
from doc_schemas import DocumentFieldExtraction


@pytest.mark.asyncio
class TestExtractPageFields:
    async def test_extracts_fields_from_a_valid_json_response(self):
        response_json = json.dumps(
            {
                "document_type": "account_closure_request",
                "applicant_name": "Jordan Rivera",
                "key_date": "2026-06-15",
                "key_amount": None,
                "notes": None,
                "confidence": "high",
                "evidence": {"applicant_name": "Applicant Name: Jordan Rivera"},
            }
        )
        runtime = FakeRuntime(default_response=response_json)
        pipeline = ExtractionPipeline(runtime, DocumentFieldExtraction)

        result = await extract_page_fields("Applicant Name: Jordan Rivera", pipeline=pipeline, model="test-model")

        assert result.parsed is not None
        assert result.parsed.applicant_name == "Jordan Rivera"
        assert result.needs_review is False

    async def test_invalid_json_flows_through_repair_and_review_queue(self):
        runtime = FakeRuntime(default_response="not json")
        pipeline = ExtractionPipeline(runtime, DocumentFieldExtraction, max_repair_attempts=1)

        result = await extract_page_fields("garbled page text", pipeline=pipeline, model="test-model")

        assert result.parsed is None
        assert result.needs_review is True
        assert len(pipeline.review_queue) == 1
