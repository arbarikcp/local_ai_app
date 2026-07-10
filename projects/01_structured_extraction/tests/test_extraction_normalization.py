import pytest

from extraction_normalization import TextTooLongError, normalize_text


class TestWhitespaceCollapsing:
    def test_collapses_multiple_spaces(self):
        assert normalize_text("A     B") == "A B"

    def test_collapses_excess_blank_lines(self):
        assert normalize_text("A\n\n\n\nB") == "A\n\nB"

    def test_trims_leading_and_trailing_whitespace(self):
        assert normalize_text("   hello world   ") == "hello world"


class TestControlCharacterStripping:
    def test_strips_null_and_control_bytes(self):
        text = "hello\x00\x01world"
        assert normalize_text(text) == "helloworld"

    def test_preserves_newlines_and_tabs(self):
        text = "line one\n\tindented line two"
        assert normalize_text(text) == "line one\n\tindented line two"


class TestLengthLimit:
    def test_text_within_the_limit_passes_through(self):
        assert normalize_text("short text", max_chars=100) == "short text"

    def test_text_exceeding_the_limit_raises(self):
        with pytest.raises(TextTooLongError) as exc_info:
            normalize_text("a" * 200, max_chars=100)
        assert exc_info.value.actual_length == 200
        assert exc_info.value.max_chars == 100

    def test_no_limit_means_no_check(self):
        assert normalize_text("a" * 10000) == "a" * 10000


class TestDoesNotRemoveLegitimateDuplicateLines:
    def test_repeated_lines_are_preserved(self):
        # Unlike Module 20's prompt_compression, normalization must not
        # drop a real, repeated line from a document (e.g. two identical
        # invoice line items).
        text = "Item: Widget - $5.00\nItem: Widget - $5.00"
        result = normalize_text(text)
        assert result.count("Item: Widget - $5.00") == 2
