import tempfile

import benchmark_and_evaluate as sut


class TestRunLab:
    async def test_produces_one_result_per_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = await sut.run_lab(tmp_dir)
        assert set(results.keys()) == {"numpy", "chroma", "lancedb"}

    async def test_metrics_are_in_valid_ranges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = await sut.run_lab(tmp_dir)
        for metrics in results.values():
            assert metrics["mean_latency_seconds"] >= 0
            for key in ("mean_recall_at_k", "mean_precision_at_k", "mrr", "mean_ndcg_at_k"):
                assert 0.0 <= metrics[key] <= 1.0

    async def test_exact_code_query_is_found_by_every_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = await sut.run_lab(tmp_dir)
        for metrics in results.values():
            assert metrics["mean_recall_at_k"] > 0.0


class TestResultsToMarkdownTable:
    async def test_includes_a_header_and_every_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = await sut.run_lab(tmp_dir)
        table = sut.results_to_markdown_table(results)
        assert "Recall@k" in table
        assert "numpy" in table
        assert "chroma" in table
        assert "lancedb" in table


class TestMain:
    def test_returns_zero_and_prints_a_comparison_table(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Recall@k" in captured.out
