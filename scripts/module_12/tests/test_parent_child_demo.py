import parent_child_demo as sut


class TestRunLab:
    async def test_ingests_all_twenty_documents(self):
        result = await sut.run_lab()
        assert result["documents"] == 20

    async def test_child_chunks_outnumber_parent_chunks(self):
        result = await sut.run_lab()
        assert result["child_chunks"] > result["parent_chunks"]

    async def test_a_parent_is_retrieved(self):
        result = await sut.run_lab()
        assert result["top_parent_id"] is not None
        assert result["top_parent_text_length"] > 0


class TestResultToMarkdown:
    async def test_includes_the_question(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert result["question"] in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "parent-child" in captured.out
