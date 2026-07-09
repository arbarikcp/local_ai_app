import multimodal_rag_demo as sut


class TestRunLab:
    async def test_loads_both_pdf_pages(self):
        result = await sut.run_lab()
        assert set(result["pdf_documents_loaded"]) == {"sample_invoice::page1", "scanned_receipt::page1"}

    async def test_only_the_digital_native_page_produces_chunks(self):
        result = await sut.run_lab()
        assert result["chunks_ingested"] == ["sample_invoice::page1::0"]

    async def test_chunk_level_citation_references_the_real_page(self):
        result = await sut.run_lab()
        assert result["chunk_level_citations"] == ["sample_invoice::page1::0"]

    async def test_source_level_citation_keeps_the_page_segment(self):
        result = await sut.run_lab()
        assert result["source_level_citations"] == ["sample_invoice::page1"]

    async def test_citations_are_grounded(self):
        result = await sut.run_lab()
        assert result["citations_are_grounded"] is True


class TestResultToMarkdown:
    async def test_includes_the_page_citation(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "sample_invoice::page1" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "page1" in captured.out
