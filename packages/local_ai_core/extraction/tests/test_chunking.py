import pytest

from local_ai_core.extraction.chunking import chunk_text, merge_partial_extractions


class TestChunkText:
    def test_short_text_returns_a_single_chunk(self):
        chunks = chunk_text("A short paragraph.", max_chars=1000)
        assert chunks == ["A short paragraph."]

    def test_rejects_nonpositive_max_chars(self):
        with pytest.raises(ValueError):
            chunk_text("text", max_chars=0)

    def test_rejects_overlap_greater_than_or_equal_to_max_chars(self):
        with pytest.raises(ValueError):
            chunk_text("text", max_chars=10, overlap_chars=10)

    def test_splits_on_paragraph_boundaries_when_possible(self):
        text = "Paragraph one is short.\n\nParagraph two is also short."
        chunks = chunk_text(text, max_chars=30)
        assert len(chunks) == 2
        assert "Paragraph one" in chunks[0]
        assert "Paragraph two" in chunks[1]

    def test_packs_multiple_short_paragraphs_into_one_chunk_when_they_fit(self):
        text = "One.\n\nTwo.\n\nThree."
        chunks = chunk_text(text, max_chars=100)
        assert len(chunks) == 1
        assert "One." in chunks[0] and "Two." in chunks[0] and "Three." in chunks[0]

    def test_every_chunk_respects_max_chars_when_no_overlap(self):
        text = "word " * 200  # long text, no natural paragraph breaks
        chunks = chunk_text(text, max_chars=50)
        assert all(len(c) <= 50 for c in chunks)

    def test_hard_splits_a_single_paragraph_longer_than_max_chars(self):
        long_paragraph = "x" * 250
        chunks = chunk_text(long_paragraph, max_chars=100)
        assert len(chunks) == 3  # 100 + 100 + 50
        assert "".join(chunks) == long_paragraph

    def test_no_content_is_lost_without_overlap(self):
        text = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota kappa lambda."
        chunks = chunk_text(text, max_chars=25)
        # every word from the original text should appear in some chunk
        original_words = set(text.replace("\n\n", " ").split())
        chunked_words = set(" ".join(chunks).split())
        assert original_words <= chunked_words

    def test_overlap_repeats_trailing_characters_in_the_next_chunk(self):
        long_paragraph = "x" * 200
        chunks = chunk_text(long_paragraph, max_chars=100, overlap_chars=20)
        assert len(chunks) >= 2
        # the last 20 chars of chunk 0 (pre-overlap) should be a prefix of chunk 1
        assert chunks[1].startswith(chunks[0][-20:])

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("", max_chars=100) == []


class TestMergePartialExtractions:
    def test_single_partial_is_returned_as_is(self):
        result = merge_partial_extractions([{"name": "Maria", "age": 29, "city": None}])
        assert result.merged == {"name": "Maria", "age": 29}
        assert result.conflicting_fields == []

    def test_fills_in_fields_missing_from_earlier_chunks(self):
        partials = [{"name": "Maria", "age": None}, {"name": None, "age": 29}]
        result = merge_partial_extractions(partials)
        assert result.merged == {"name": "Maria", "age": 29}

    def test_first_non_null_value_wins_when_no_conflict(self):
        partials = [{"city": "Austin"}, {"city": "Austin"}]
        result = merge_partial_extractions(partials)
        assert result.merged == {"city": "Austin"}
        assert result.conflicting_fields == []

    def test_flags_conflicting_nonnull_values_instead_of_silently_picking_one(self):
        partials = [{"city": "Austin"}, {"city": "Denver"}]
        result = merge_partial_extractions(partials)
        assert result.conflicting_fields == ["city"]
        assert result.merged["city"] == "Austin"  # first value kept, but flagged

    def test_all_null_field_is_absent_from_merged_result(self):
        partials = [{"city": None}, {"city": None}]
        result = merge_partial_extractions(partials)
        assert "city" not in result.merged

    def test_empty_partials_list_returns_empty_result(self):
        result = merge_partial_extractions([])
        assert result.merged == {}
        assert result.conflicting_fields == []
