from dataclasses import dataclass

from doc_prompts import build_doc_context, build_doc_qa_prompt, extract_page_citations


@dataclass(frozen=True)
class FakePage:
    page_id: str
    extracted_text: str


class TestBuildDocContext:
    def test_tags_each_page_with_its_page_id(self):
        pages = [FakePage("multi_page_form::page1", "Applicant Name: Jordan Rivera")]
        context = build_doc_context(pages)
        assert context == "[multi_page_form::page1] Applicant Name: Jordan Rivera"

    def test_joins_multiple_pages_with_a_blank_line(self):
        pages = [
            FakePage("multi_page_form::page1", "text one"),
            FakePage("multi_page_form::page2", "text two"),
        ]
        context = build_doc_context(pages)
        assert context == "[multi_page_form::page1] text one\n\n[multi_page_form::page2] text two"


class TestBuildDocQaPrompt:
    def test_embeds_question_and_context(self):
        pages = [FakePage("multi_page_form::page2", "Refund Amount Owed: $42.50")]
        prompt = build_doc_qa_prompt("What is the refund amount?", pages)
        assert "What is the refund amount?" in prompt
        assert "[multi_page_form::page2] Refund Amount Owed: $42.50" in prompt
        assert "Cite every page you use" in prompt


class TestExtractPageCitations:
    def test_extracts_a_page_id_shaped_citation(self):
        answer = "The refund amount owed is $42.50 [multi_page_form::page2]."
        assert extract_page_citations(answer) == ["multi_page_form::page2"]

    def test_extracts_multiple_unique_citations_in_first_seen_order(self):
        answer = "Per [doc::page1] and [doc::page2], and again [doc::page1]."
        assert extract_page_citations(answer) == ["doc::page1", "doc::page2"]

    def test_does_not_match_a_chunk_index_shaped_citation(self):
        # citation_packer's own chunk-shaped markers (`doc::0`) are a
        # different convention and should not be picked up here.
        answer = "See [nimbus_handbook::0] for details."
        assert extract_page_citations(answer) == []

    def test_no_citations_returns_empty_list(self):
        assert extract_page_citations("I don't know based on the provided document.") == []
