import tempfile

import store_comparison as sut


class TestRunLab:
    async def test_all_backends_agree_on_the_top_result(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await sut.run_lab(tmp_dir)
        assert result["agree_across_backends"] is True

    async def test_metadata_filter_returns_the_same_ids_across_backends(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await sut.run_lab(tmp_dir)
        filtered = result["filtered_by_store"]
        assert set(filtered["numpy"]) == set(filtered["chroma"]) == set(filtered["lancedb"])

    async def test_hybrid_search_recovers_the_exact_code_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await sut.run_lab(tmp_dir)
        assert "doc_order_code" in result["hybrid_result_ids"]


class TestResultToMarkdown:
    async def test_includes_backend_names_and_agreement_verdict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await sut.run_lab(tmp_dir)
        markdown = sut.result_to_markdown(result)
        assert "numpy" in markdown
        assert "chroma" in markdown
        assert "lancedb" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "store comparison" in captured.out
