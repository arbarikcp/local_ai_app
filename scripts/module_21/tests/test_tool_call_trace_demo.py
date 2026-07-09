import tool_call_trace_demo as sut


class TestRunLab:
    def test_records_both_tool_calls(self):
        result = sut.run_lab()
        assert len(result["tool_call_spans"]) == 2

    def test_metrics_and_trace_agree_on_counts(self):
        result = sut.run_lab()
        assert result["tool_call_count"] == 2
        assert result["tool_error_count"] == 1

    def test_the_failing_call_carries_its_error(self):
        result = sut.run_lab()
        failing = [s for s in result["tool_call_spans"] if s["tool_name"] == "cancel_order"][0]
        assert failing["error"] == "order not found"

    def test_the_succeeding_call_has_no_error(self):
        result = sut.run_lab()
        succeeding = [s for s in result["tool_call_spans"] if s["tool_name"] == "lookup_order"][0]
        assert succeeding["error"] is None


class TestResultToMarkdown:
    def test_markdown_marks_the_failing_call(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "error" in markdown
        assert "ok" in markdown
