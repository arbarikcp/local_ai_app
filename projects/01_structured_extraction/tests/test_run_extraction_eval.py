import run_extraction_eval as sut


class TestLoadDataset:
    def test_loads_the_real_committed_dataset(self):
        examples = sut.load_dataset(sut.DATASET_PATH)
        assert len(examples) == 10
        assert {e.schema_name for e in examples} == {"invoice_v1", "support_ticket_v1"}


class TestRunLab:
    async def test_perfect_scenario_scores_a_flawless_run(self):
        result = await sut.run_lab()
        perfect = result["perfect"]
        assert perfect.mean_field_exact_match == 1.0
        assert perfect.invalid_json_rate == 0.0
        assert perfect.review_rate == 0.0

    async def test_imperfect_scenario_catches_the_deliberately_broken_examples(self):
        result = await sut.run_lab()
        imperfect = result["imperfect"]
        assert imperfect.invalid_json_rate == 0.2
        assert imperfect.review_rate == 0.2
        assert imperfect.mean_field_exact_match < 1.0

    async def test_imperfect_scenario_is_never_better_than_perfect(self):
        result = await sut.run_lab()
        assert result["imperfect"].mean_field_exact_match <= result["perfect"].mean_field_exact_match
        assert result["imperfect"].invalid_json_rate >= result["perfect"].invalid_json_rate


class TestResultToMarkdown:
    async def test_markdown_reports_both_scenarios(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "perfect" in markdown
        assert "imperfect" in markdown
        assert "100.00%" in markdown
