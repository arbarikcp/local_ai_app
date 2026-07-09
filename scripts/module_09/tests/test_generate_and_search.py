import generate_and_search as sut

from local_ai_rag.embeddings.fake import FakeEmbedder


class TestBuildIndex:
    async def test_indexes_every_document_in_the_corpus(self):
        embedder = FakeEmbedder(dimensions=32)
        index = await sut.build_index(embedder)
        assert len(index) == len(sut.CORPUS)

    async def test_indexed_metadata_matches_the_corpus(self):
        embedder = FakeEmbedder(dimensions=32)
        index = await sut.build_index(embedder)
        results = index.search(await embedder.embed_query("password"), k=len(sut.CORPUS))
        by_id = {r.doc_id: r for r in results}
        assert by_id["doc_password"].metadata == {"category": "account"}


class TestRunLab:
    async def test_top_result_for_password_query_is_a_password_doc(self):
        embedder = FakeEmbedder(dimensions=64)
        result = await sut.run_lab(embedder, k=3)
        assert result["top_result_doc_id"] in {"doc_password", "doc_password2"}

    async def test_metadata_filter_only_returns_account_docs(self):
        embedder = FakeEmbedder(dimensions=64)
        result = await sut.run_lab(embedder, k=5)
        assert set(result["filtered_result_doc_ids"]) <= {"doc_password", "doc_password2"}

    async def test_eval_metrics_are_in_valid_ranges(self):
        embedder = FakeEmbedder(dimensions=64)
        result = await sut.run_lab(embedder, k=3)
        for key in ("mean_recall_at_k", "mean_precision_at_k", "mrr", "mean_ndcg_at_k"):
            assert 0.0 <= result[key] <= 1.0

    async def test_index_size_matches_corpus_size(self):
        embedder = FakeEmbedder(dimensions=32)
        result = await sut.run_lab(embedder, k=3)
        assert result["index_size"] == len(sut.CORPUS)


class TestResultToMarkdown:
    async def test_includes_the_query_and_top_result(self):
        embedder = FakeEmbedder(dimensions=32)
        result = await sut.run_lab(embedder, k=3)
        markdown = sut.result_to_markdown(result)
        assert result["query"] in markdown
        assert result["top_result_doc_id"] in markdown

    def test_handles_a_missing_top_result_without_crashing(self):
        result = {
            "index_size": 0,
            "query": "q",
            "top_result_doc_id": None,
            "top_result_score": None,
            "filtered_result_doc_ids": [],
            "mean_recall_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mrr": 0.0,
            "mean_ndcg_at_k": 0.0,
        }
        markdown = sut.result_to_markdown(result)
        assert "n/a" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Labs 1-4, 6" in captured.out

    def test_respects_the_k_argument(self, capsys):
        exit_code = sut.main(["--k", "1"])
        assert exit_code == 0
