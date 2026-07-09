import context_budget_demo as sut


class TestRunLab:
    def test_remaining_for_retrieval_is_history_budget_minus_history_used(self):
        result = sut.run_lab()
        assert result["remaining_for_retrieval"] == (
            result["conversation_history_budget"] - result["conversation_history_tokens_used"]
        )

    def test_both_candidate_chunks_fit_within_the_generous_remaining_budget(self):
        result = sut.run_lab()
        assert result["packed_chunk_count"] == 2
        assert "password-reset-guide" in result["packed_doc_ids"]


class TestResultToMarkdown:
    def test_markdown_reports_the_context_window(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "4096" in markdown
