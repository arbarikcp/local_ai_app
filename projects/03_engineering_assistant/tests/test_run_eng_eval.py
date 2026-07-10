import run_eng_eval as sut


class TestLoadIntentGoldenSet:
    def test_loads_the_real_committed_golden_set(self):
        cases = sut.load_intent_golden_set(sut.GOLDEN_SET_PATH)
        assert len(cases) == 16


class TestRunIntentEval:
    def test_accuracy_is_a_real_fraction_not_perfect(self):
        cases = sut.load_intent_golden_set(sut.GOLDEN_SET_PATH)
        summary = sut.run_intent_eval(cases)
        # A real, honest keyword-classifier limitation: "Execute the test
        # suite." doesn't literally contain "execute test" - accuracy
        # should reflect that miss, not be silently perfect.
        assert 0.0 < summary.accuracy < 1.0


class TestRunLab:
    async def test_all_six_failure_cases_are_caught(self):
        summary = await sut.run_lab()
        assert summary.failure_cases_total == 6
        assert summary.failure_cases_caught == 6

    async def test_every_failure_case_detail_reports_caught(self):
        summary = await sut.run_lab()
        assert all(r.caught for r in summary.failure_case_details)

    async def test_the_happy_path_is_verified(self):
        summary = await sut.run_lab()
        assert summary.happy_path_verified is True

    async def test_intent_accuracy_is_included(self):
        summary = await sut.run_lab()
        assert 0.0 < summary.intent_accuracy <= 1.0


class TestSummaryToMarkdown:
    async def test_markdown_reports_every_failure_case_by_name(self):
        summary = await sut.run_lab()
        markdown = sut.summary_to_markdown(summary)
        for name in [
            "invented_file_path", "unrelated_file_change", "unsafe_shell_command",
            "invalid_patch", "missing_dependency_import", "tests_that_do_not_run",
        ]:
            assert name in markdown
