import model_router_demo as sut


class TestRunLab:
    async def test_routes_each_request_to_the_expected_tier(self):
        results = await sut.run_lab()
        by_label = {r["label"]: r for r in results}
        assert by_label["simple classification"]["tier"] == "small"
        assert by_label["long document summary"]["tier"] == "large"
        assert by_label["multi-step agent plan"]["tier"] == "large"
        assert by_label["tool-calling request"]["tier"] == "large"

    async def test_the_response_comes_from_the_matching_tier_runtime(self):
        results = await sut.run_lab()
        by_label = {r["label"]: r for r in results}
        assert "(small model)" in by_label["simple classification"]["response"]
        assert "(large model)" in by_label["multi-step agent plan"]["response"]

    async def test_run_lab_is_repeatable(self):
        # REQUESTS is module-level state - a naive implementation could
        # mutate it on the first call and break the second.
        first = await sut.run_lab()
        second = await sut.run_lab()
        assert [r["label"] for r in first] == [r["label"] for r in second]


class TestResultsToMarkdown:
    async def test_markdown_lists_every_request(self):
        results = await sut.run_lab()
        markdown = sut.results_to_markdown(results)
        assert "simple classification" in markdown
        assert "tool-calling request" in markdown
