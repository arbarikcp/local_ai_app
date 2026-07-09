import compare_chunk_sizes as sut


class TestCompareChunkSizes:
    async def test_produces_one_result_per_chunk_size(self):
        results = await sut.compare_chunk_sizes([200, 800], k=3)
        assert set(results.keys()) == {200, 800}

    async def test_smaller_chunk_size_produces_more_chunks(self):
        results = await sut.compare_chunk_sizes([200, 1200], k=3)
        assert results[200]["chunks_in_index"] > results[1200]["chunks_in_index"]


class TestResultsToMarkdownTable:
    async def test_includes_every_chunk_size(self):
        results = await sut.compare_chunk_sizes([300], k=3)
        table = sut.results_to_markdown_table(results)
        assert "300" in table
        assert "Recall@k" in table


class TestMain:
    def test_returns_zero_and_prints_a_comparison_table(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "chunk size" in captured.out.lower()
