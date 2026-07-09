from local_ai_rag.context_packers.citation_packer import (
    build_context,
    build_rag_prompt,
    extract_citations,
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
