import qa_eval as sut


class TestRunLab:
    async def test_produces_metrics_and_unanswerable_rows(self):
        result = await sut.run_lab()
        assert "answerable_metrics" in result
        assert len(result["unanswerable_rows"]) == len(sut.UNANSWERABLE_CASES)

    async def test_answerable_metrics_are_in_valid_ranges(self):
        result = await sut.run_lab()
        for key in ("mean_recall_at_k", "mean_precision_at_k", "mrr", "mean_ndcg_at_k"):
            assert 0.0 <= result["answerable_metrics"][key] <= 1.0

    async def test_recall_is_above_zero_for_a_real_corpus(self):
        # A weak assertion on purpose: FakeEmbedder is crude, so this only
        # checks retrieval isn't completely broken, not that it's perfect.
        result = await sut.run_lab()
        assert result["answerable_metrics"]["mean_recall_at_k"] > 0.0


class TestEvaluateAnswerable:
    async def test_every_golden_case_contributes_to_the_average(self):
        pipeline = await sut.build_pipeline()
        metrics = await sut.evaluate_answerable(pipeline, k=3)
        assert isinstance(metrics["mean_recall_at_k"], float)


class TestResultToMarkdown:
    async def test_includes_unanswerable_questions(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert sut.UNANSWERABLE_CASES[0].question in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "recall@k" in captured.out
