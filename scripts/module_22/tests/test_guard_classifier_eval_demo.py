import guard_classifier_eval_demo as sut


class TestRunLab:
    def test_catch_rate_is_high_but_not_necessarily_perfect(self):
        result = sut.run_lab()
        assert result["catch_rate"] > 0.9

    def test_false_positive_rate_is_zero_on_this_dataset(self):
        result = sut.run_lab()
        assert result["false_positive_rate"] == 0.0

    def test_counts_are_internally_consistent(self):
        result = sut.run_lab()
        assert result["true_positives"] + result["false_negatives"] == 25
        assert result["true_negatives"] + result["false_positives"] == 14

    def test_the_known_underscore_filename_gap_is_a_documented_false_negative(self):
        result = sut.run_lab()
        assert any("ignore_all_previous_instructions" in text for text in result["false_negative_examples"])


class TestResultToMarkdown:
    def test_markdown_reports_the_catch_rate(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Catch rate" in markdown
