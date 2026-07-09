import pdf_extraction_demo as sut


class TestRunLab:
    def test_extracts_real_invoice_text(self):
        result = sut.run_lab()
        assert "Invoice" in result["extracted_text"]
        assert "INV-2026-0042" in result["extracted_text"]

    def test_extracts_the_real_table_header(self):
        result = sut.run_lab()
        assert result["table"][0] == ["Item", "Qty", "Unit Price"]

    def test_extracts_a_positive_word_count(self):
        result = sut.run_lab()
        assert result["word_count"] > 0

    def test_renders_a_real_sized_image(self):
        result = sut.run_lab()
        width, height = result["rendered_image_size"]
        assert width > 0 and height > 0


class TestResultToMarkdown:
    def test_includes_the_extracted_text(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Invoice" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "table" in captured.out.lower()
