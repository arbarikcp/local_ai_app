import observability_dashboard_demo as sut


class TestRunLab:
    def test_counts_every_request(self):
        result = sut.run_lab()
        assert result["request_count"] == 3

    def test_feedback_summary_matches_the_scripted_ratings(self):
        result = sut.run_lab()
        assert result["feedback_summary"] == {"up": 2, "down": 1}

    def test_no_trace_is_missing_a_core_step(self):
        result = sut.run_lab()
        assert result["incomplete_traces"] == []

    def test_mean_latency_is_a_real_nonnegative_number(self):
        result = sut.run_lab()
        assert result["mean_latency_ms"] >= 0


class TestResultToMarkdown:
    def test_markdown_reports_the_feedback_summary(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "up" in markdown
        assert "down" in markdown
