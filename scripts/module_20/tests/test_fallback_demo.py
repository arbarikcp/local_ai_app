import fallback_demo as sut


class TestRunLab:
    async def test_falls_back_to_the_secondary_runtime(self):
        result = await sut.run_lab()
        assert result["fallback_runtime_index"] == 1
        assert result["fallback_attempts"] == 2
        assert "fallback runtime" in result["fallback_response"]

    async def test_non_retryable_error_propagates_without_calling_the_next_runtime(self):
        result = await sut.run_lab()
        assert result["validation_error_propagated_without_fallback"] is True
        assert result["never_called_call_count"] == 0

    async def test_every_runtime_down_raises(self):
        result = await sut.run_lab()
        assert result["all_runtimes_down_raised"] is True


class TestResultToMarkdown:
    async def test_markdown_mentions_the_fallback_outcome(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "NoRuntimesAvailable" in markdown or "raised" in markdown
