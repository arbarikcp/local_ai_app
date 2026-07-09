import incremental_indexing_demo as sut


class TestRunLab:
    async def test_first_sync_ingests_all_documents(self):
        result = await sut.run_lab()
        assert result["documents_first_sync"] == 20

    async def test_second_sync_only_updates_the_edited_document(self):
        result = await sut.run_lab()
        assert result["second_sync_updated"] == ["password_reset"]

    async def test_second_sync_deletes_the_removed_document(self):
        result = await sut.run_lab()
        assert result["second_sync_deleted"] == ["supported_regions"]

    async def test_second_sync_does_not_re_embed_unchanged_documents(self):
        result = await sut.run_lab()
        # Only the one updated document should trigger an embed_documents() call.
        assert result["embed_calls_for_second_sync"] == 1

    async def test_eighteen_documents_are_unchanged(self):
        result = await sut.run_lab()
        assert result["second_sync_unchanged_count"] == 18


class TestResultToMarkdown:
    async def test_includes_the_diff_summary(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "password_reset" in markdown
        assert "supported_regions" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "incremental indexing" in captured.out
