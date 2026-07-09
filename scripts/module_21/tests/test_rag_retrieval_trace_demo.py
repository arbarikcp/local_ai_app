import rag_retrieval_trace_demo as sut


class TestRunLab:
    def test_retrieves_the_closest_document_first(self):
        result = sut.run_lab()
        assert result["retrieved_doc_ids"][0] == "password-reset-guide"

    def test_top_score_is_a_real_high_cosine_similarity(self):
        result = sut.run_lab()
        assert result["top_score"] > 0.9

    def test_retrieval_trace_spans_are_recorded(self):
        result = sut.run_lab()
        assert "retrieval_query" in result["span_names"]
        assert "retrieved_chunk_ids" in result["span_names"]
        assert "reranker_scores" in result["span_names"]


class TestResultToMarkdown:
    def test_markdown_reports_the_top_result(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "password-reset-guide" in markdown
