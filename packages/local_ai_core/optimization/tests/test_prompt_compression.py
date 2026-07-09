from local_ai_core.optimization.prompt_compression import compress_prompt


class TestDuplicateLineRemoval:
    def test_drops_a_consecutive_duplicate_line(self):
        text = "Please answer the question.\nPlease answer the question.\nWhat is 2+2?"
        result = compress_prompt(text)
        assert result.compressed_text.count("Please answer the question.") == 1

    def test_does_not_drop_non_consecutive_duplicate_lines(self):
        text = "A\nB\nA"
        result = compress_prompt(text)
        assert result.compressed_text.count("A") == 2

    def test_does_not_collapse_duplicate_blank_lines_into_one_blank_line(self):
        text = "A\n\nB"
        result = compress_prompt(text)
        assert "A\n\nB" in result.compressed_text


class TestWhitespaceCollapsing:
    def test_collapses_three_or_more_blank_lines_to_one(self):
        text = "A\n\n\n\nB"
        result = compress_prompt(text)
        assert result.compressed_text == "A\n\nB"

    def test_collapses_multiple_spaces_to_one(self):
        text = "A     B"
        result = compress_prompt(text)
        assert result.compressed_text == "A B"


class TestReductionIsGenuine:
    def test_duplicate_heavy_text_produces_a_positive_reduction_ratio(self):
        text = "\n".join(["Repeat this line."] * 5 + ["Unique final line."])
        result = compress_prompt(text)
        assert result.compressed_token_estimate < result.original_token_estimate
        assert result.reduction_ratio > 0

    def test_already_clean_text_has_zero_reduction(self):
        text = "A short, clean prompt with no duplication."
        result = compress_prompt(text)
        assert result.reduction_ratio == 0.0

    def test_empty_text_has_zero_reduction_ratio(self):
        result = compress_prompt("")
        assert result.reduction_ratio == 0.0
