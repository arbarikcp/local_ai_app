import run_rag_evaluation as sut


class TestRunLab:
    async def test_evaluates_every_golden_case(self):
        rows = await sut.run_lab()
        assert len(rows) == 16

    async def test_corrupted_invented_citation_case_is_flagged(self):
        rows = await sut.run_lab()
        row = next(r for r in rows if r["question_id"] == "q_003")
        assert row["citations_are_grounded"] is False

    async def test_corrupted_refusal_case_is_flagged(self):
        rows = await sut.run_lab()
        row = next(r for r in rows if r["question_id"] == "q_016")
        assert row["refused"] is False

    async def test_other_unanswerable_cases_correctly_refuse(self):
        rows = await sut.run_lab()
        others = [r for r in rows if r["requires_refusal"] and r["question_id"] != "q_016"]
        assert all(r["refused"] is True for r in others)

    async def test_answerable_cases_have_retrieval_metrics_populated(self):
        rows = await sut.run_lab()
        answerable = [r for r in rows if not r["requires_refusal"]]
        assert all(r["context_precision"] is not None for r in answerable)

    async def test_unanswerable_cases_have_no_retrieval_metrics(self):
        rows = await sut.run_lab()
        unanswerable = [r for r in rows if r["requires_refusal"]]
        assert all(r["context_precision"] is None for r in unanswerable)


class TestSummarize:
    async def test_metrics_are_in_valid_ranges(self):
        rows = await sut.run_lab()
        summary = sut.summarize(rows)
        for key in ("mean_context_precision", "mean_context_recall", "mean_reciprocal_rank"):
            assert 0.0 <= summary[key] <= 1.0


class TestResultToMarkdown:
    async def test_includes_the_failure_lists(self):
        rows = await sut.run_lab()
        markdown = sut.result_to_markdown(rows)
        assert "q_003" in markdown
        assert "q_016" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "context precision" in captured.out
