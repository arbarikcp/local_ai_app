import build_and_query as sut


class TestRunLab:
    async def test_ingests_all_twenty_documents(self):
        result = await sut.run_lab()
        assert result["documents_ingested"] == 20

    async def test_top_retrieved_chunk_is_from_the_password_reset_doc(self):
        result = await sut.run_lab()
        assert result["retrieved_chunk_ids"][0].startswith("password_reset::")

    async def test_citations_are_grounded(self):
        result = await sut.run_lab()
        assert result["citations_are_grounded"] is True


class TestResultToMarkdown:
    async def test_includes_the_question_and_answer(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert result["question"] in markdown
        assert result["answer_text"] in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "naive RAG" in captured.out
