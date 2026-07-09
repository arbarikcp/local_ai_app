import reranking_demo as sut


class TestRunLab:
    async def test_low_clearance_excludes_the_restricted_document(self):
        result = await sut.run_lab()
        assert "security_incident_response" not in result["low_clearance_packed_doc_ids"]

    async def test_high_clearance_can_include_the_restricted_document(self):
        result = await sut.run_lab()
        assert "security_incident_response" in result["high_clearance_packed_doc_ids"]

    async def test_a_citation_to_an_acl_filtered_document_is_flagged_ungrounded(self):
        result = await sut.run_lab()
        assert result["low_clearance_citations_grounded"] is False

    async def test_tight_budget_packs_no_more_chunks_than_a_generous_one(self):
        result = await sut.run_lab()
        assert result["tight_budget_chunks_packed"] <= result["generous_budget_chunks_packed"]


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
        assert "ACL filtering" in captured.out
