import vlm_routing_demo as sut


class TestRunLab:
    async def test_the_invoice_routes_to_the_text_llm(self):
        result = await sut.run_lab()
        assert result["invoice_result"]["route"] == "text_llm"

    async def test_the_scanned_receipt_routes_to_the_vlm(self):
        result = await sut.run_lab()
        assert result["receipt_result"]["route"] == "vlm"

    async def test_the_invoice_answer_comes_from_the_text_runtime(self):
        result = await sut.run_lab()
        assert "invoice total" in result["invoice_result"]["answer"].lower()

    async def test_the_receipt_answer_comes_from_the_vlm(self):
        result = await sut.run_lab()
        assert "receipt" in result["receipt_result"]["answer"].lower()

    async def test_screenshot_answer_is_produced_by_the_vlm(self):
        result = await sut.run_lab()
        assert result["screenshot_answer"]


class TestResultToMarkdown:
    async def test_includes_both_routing_decisions(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "text_llm" in markdown
        assert "vlm" in markdown


class TestMain:
    def test_returns_zero_and_prints_a_report(self, capsys):
        exit_code = sut.main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "routed to" in captured.out
