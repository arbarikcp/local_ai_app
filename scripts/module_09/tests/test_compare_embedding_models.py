import compare_embedding_models as sut

from local_ai_rag.embeddings.fake import FakeEmbedder


class TestCompareEmbedders:
    async def test_produces_one_summary_per_embedder(self):
        embedders = {"a": FakeEmbedder(dimensions=64), "b": FakeEmbedder(dimensions=4)}
        results = await sut.compare_embedders(embedders, k=3)
        assert set(results.keys()) == {"a", "b"}

    async def test_higher_dimensional_embedder_has_no_lower_ndcg_than_the_collision_prone_one(self):
        embedders = {"high-d": FakeEmbedder(dimensions=64), "low-d": FakeEmbedder(dimensions=4)}
        results = await sut.compare_embedders(embedders, k=3)
        assert results["high-d"].mean_ndcg_at_k >= results["low-d"].mean_ndcg_at_k


class TestResultsToMarkdownTable:
    async def test_includes_every_model_name_and_a_header_row(self):
        embedders = {"model-x": FakeEmbedder(dimensions=32)}
        results = await sut.compare_embedders(embedders, k=3)
        table = sut.results_to_markdown_table(results)
        assert "model-x" in table
        assert "Recall@k" in table


class TestMain:
    def test_returns_zero_and_prints_a_comparison_table(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "fake-64d" in captured.out
        assert "fake-4d" in captured.out
