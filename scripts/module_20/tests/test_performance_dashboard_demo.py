import performance_dashboard_demo as sut


class TestRunLab:
    async def test_counts_every_request_including_failures(self):
        result = await sut.run_lab()
        assert result["request_count"] == 10
        assert result["error_count"] == 2

    async def test_error_rate_matches_the_scripted_failure_ratio(self):
        result = await sut.run_lab()
        assert result["error_rate"] == 0.2

    async def test_p95_is_never_less_than_p50(self):
        result = await sut.run_lab()
        assert result["p95_latency_ms"] >= result["p50_latency_ms"]

    async def test_mean_tokens_per_second_is_positive(self):
        result = await sut.run_lab()
        assert result["mean_tokens_per_second"] > 0


class TestResultToMarkdown:
    async def test_markdown_reports_the_error_rate(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "20%" in markdown
