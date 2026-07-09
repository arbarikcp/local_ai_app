from local_ai_rag.context_packers.citation_packer import (
    build_context,
    build_rag_prompt,
    extract_citations,
    summarize_source_citations,
)
from local_ai_rag.embeddings.embedder import SearchResult


def make_result(doc_id: str, text: str) -> SearchResult:
    return SearchResult(doc_id=doc_id, score=0.9, text=text, metadata={})


class TestBuildContext:
    def test_tags_each_chunk_with_its_citation_key(self):
        context = build_context([make_result("password_reset::0", "reset link expires in 15 minutes")])
        assert "[password_reset::0]" in context
        assert "reset link expires in 15 minutes" in context

    def test_joins_multiple_chunks(self):
        context = build_context([make_result("a::0", "first"), make_result("b::0", "second")])
        assert "[a::0] first" in context
        assert "[b::0] second" in context

    def test_empty_results_produces_empty_context(self):
        assert build_context([]) == ""


class TestBuildRagPrompt:
    def test_matches_the_curriculum_minimal_rag_prompt_shape(self):
        prompt = build_rag_prompt("How long until my reset link expires?", [make_result("a::0", "15 minutes")])
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt
        assert "How long until my reset link expires?" in prompt
        assert "I don't know based on the provided documents." in prompt

    def test_embeds_the_built_context(self):
        results = [make_result("a::0", "unique marker text")]
        prompt = build_rag_prompt("q", results)
        assert "[a::0] unique marker text" in prompt


class TestExtractCitations:
    def test_finds_a_single_citation(self):
        assert extract_citations("The link expires in 15 minutes [password_reset::0].") == ["password_reset::0"]

    def test_finds_multiple_unique_citations_in_order(self):
        text = "See [a::0] and also [b::1], and again [a::0]."
        assert extract_citations(text) == ["a::0", "b::1"]

    def test_no_citations_returns_empty_list(self):
        assert extract_citations("No citations here.") == []

    def test_ignores_bracketed_text_that_is_not_chunk_id_shaped(self):
        assert extract_citations("See [not a citation] for details.") == []

    def test_finds_a_citation_with_multiple_colon_separated_segments(self):
        # Module 18's PDF-page doc_ids (pdf_stem::pageN) add a second "::"
        # before the chunk index - a real bug this exact case caught.
        text = "The invoice number is [sample_invoice::page1::0]."
        assert extract_citations(text) == ["sample_invoice::page1::0"]


class TestSummarizeSourceCitations:
    def test_reduces_chunk_citations_to_unique_doc_ids(self):
        result = summarize_source_citations(["password_reset::0", "password_reset::1"])
        assert result == ["password_reset"]

    def test_preserves_first_seen_order_across_documents(self):
        result = summarize_source_citations(["b::0", "a::0", "b::1"])
        assert result == ["b", "a"]

    def test_empty_input_returns_empty_list(self):
        assert summarize_source_citations([]) == []

    def test_a_pdf_page_source_keeps_its_page_segment(self):
        # rsplit, not split - only the trailing "::chunk_index" is
        # stripped, so a page-qualified doc_id doesn't collapse to just
        # the PDF's stem and lose which page the citation came from.
        result = summarize_source_citations(["sample_invoice::page1::0", "sample_invoice::page2::0"])
        assert result == ["sample_invoice::page1", "sample_invoice::page2"]
