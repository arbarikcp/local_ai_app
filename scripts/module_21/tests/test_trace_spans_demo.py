import trace_spans_demo as sut


class TestRunLab:
    def test_no_core_steps_are_missing(self):
        result = sut.run_lab()
        assert result["missing_core_steps"] == []

    def test_span_order_matches_the_curriculum_trace_model(self):
        result = sut.run_lab()
        assert result["span_names"] == [
            "input_validation",
            "prompt_template_version",
            "retrieval_query",
            "retrieved_chunk_ids",
            "reranker_scores",
            "context_packing",
            "model_call",
            "output_validation",
            "final_response",
            "evaluation_hooks",
        ]

    def test_total_elapsed_reflects_the_real_sleeps(self):
        result = sut.run_lab()
        assert result["total_elapsed_ms"] >= 3  # two sleeps of 1ms + 2ms


class TestResultToMarkdown:
    def test_markdown_shows_the_span_order(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "model_call" in markdown
        assert "->" in markdown
