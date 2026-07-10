import pytest
from local_ai_core.runtimes.fake import FakeRuntime

from doc_qa import answer_question
from doc_storage import PageAnalysisRecord


def _page(page_id, page_number, text, route="text_llm"):
    return PageAnalysisRecord(
        page_id=page_id,
        doc_id="multi_page_form",
        page_number=page_number,
        route=route,
        route_reason="reason",
        extracted_text=text,
    )


@pytest.mark.asyncio
class TestAnswerQuestion:
    async def test_a_grounded_citation_is_verified(self):
        pages = [
            _page("multi_page_form::page1", 1, "Applicant Name: Jordan Rivera"),
            _page("multi_page_form::page2", 2, "Refund Amount Owed: $42.50"),
        ]
        runtime = FakeRuntime(
            default_response="The refund amount owed is $42.50 [multi_page_form::page2]."
        )

        result = await answer_question(pages, "What is the refund amount?", runtime=runtime, model="test-model")

        assert len(result.citations) == 1
        assert result.citations[0].page_id == "multi_page_form::page2"
        assert result.citations[0].page_number == 2
        assert result.citations[0].verified is True
        assert result.latency_ms >= 0

    async def test_an_invented_citation_is_not_verified(self):
        pages = [_page("multi_page_form::page1", 1, "Applicant Name: Jordan Rivera")]
        runtime = FakeRuntime(default_response="Per [multi_page_form::page9], the answer is X.")

        result = await answer_question(pages, "some question", runtime=runtime, model="test-model")

        assert len(result.citations) == 1
        assert result.citations[0].verified is False
        assert result.citations[0].page_number == -1

    async def test_quarantined_pages_are_excluded_from_context_and_cannot_be_cited_valid(self):
        pages = [
            _page("multi_page_form::page1", 1, "Applicant Name: Jordan Rivera"),
            _page("multi_page_form::page2", 2, "", route="quarantined"),
        ]
        runtime = FakeRuntime(default_response="See [multi_page_form::page2] for details.")

        result = await answer_question(pages, "some question", runtime=runtime, model="test-model")

        prompt_sent = runtime.requests_received[0].prompt
        assert "multi_page_form::page2" not in prompt_sent
        assert result.citations[0].verified is False

    async def test_no_citations_in_the_answer_returns_an_empty_citation_list(self):
        pages = [_page("multi_page_form::page1", 1, "Applicant Name: Jordan Rivera")]
        runtime = FakeRuntime(default_response="I don't know based on the provided document.")

        result = await answer_question(pages, "some question", runtime=runtime, model="test-model")

        assert result.citations == []
