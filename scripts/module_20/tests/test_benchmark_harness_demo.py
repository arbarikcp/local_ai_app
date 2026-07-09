import benchmark_harness_demo as sut


class TestRunLab:
    async def test_returns_one_result_per_config(self):
        results = await sut.run_lab()
        assert len(results) == 3
        assert {r.name for r in results} == {"q4_small_model", "q8_medium_model", "fp16_large_model"}

    async def test_faster_configured_latency_produces_lower_mean_latency(self):
        results = await sut.run_lab()
        by_name = {r.name: r for r in results}
        assert by_name["q4_small_model"].mean_latency_ms < by_name["q8_medium_model"].mean_latency_ms
        assert by_name["q8_medium_model"].mean_latency_ms < by_name["fp16_large_model"].mean_latency_ms


class TestResultsToMarkdown:
    async def test_markdown_lists_every_config(self):
        results = await sut.run_lab()
        markdown = sut.results_to_markdown(results)
        assert "q4_small_model" in markdown
        assert "fp16_large_model" in markdown
