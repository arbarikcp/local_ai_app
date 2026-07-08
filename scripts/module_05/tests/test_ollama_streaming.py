import json

import pytest

from ollama_streaming import (
    StreamChunk,
    accumulate_stream,
    iter_stream_chunks,
    parse_stream_line,
)

# Fixture lines shaped like real Ollama /api/generate stream=True output.
FIXTURE_LINES = [
    json.dumps({"model": "qwen2.5:1.5b", "response": "Hello", "done": False}),
    json.dumps({"model": "qwen2.5:1.5b", "response": ", world", "done": False}),
    json.dumps({"model": "qwen2.5:1.5b", "response": "!", "done": False}),
    json.dumps(
        {
            "model": "qwen2.5:1.5b",
            "response": "",
            "done": True,
            "prompt_eval_count": 8,
            "eval_count": 3,
            "eval_duration": 300_000_000,
        }
    ),
]


class TestParseStreamLine:
    def test_parses_a_normal_chunk(self):
        chunk = parse_stream_line(FIXTURE_LINES[0])
        assert chunk is not None
        assert chunk.text == "Hello"
        assert chunk.done is False

    def test_parses_the_final_done_chunk_with_usage_fields(self):
        chunk = parse_stream_line(FIXTURE_LINES[-1])
        assert chunk is not None
        assert chunk.done is True
        assert chunk.eval_count == 3
        assert chunk.eval_duration_ns == 300_000_000

    def test_returns_none_for_blank_line(self):
        assert parse_stream_line("") is None
        assert parse_stream_line("   \n") is None

    def test_raises_on_malformed_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_stream_line("not json")

    def test_defaults_missing_optional_fields_to_none(self):
        chunk = parse_stream_line(json.dumps({"response": "x", "done": False}))
        assert chunk.eval_count is None
        assert chunk.eval_duration_ns is None


class TestIterStreamChunks:
    def test_yields_one_chunk_per_nonblank_line(self):
        chunks = list(iter_stream_chunks(FIXTURE_LINES))
        assert len(chunks) == 4
        assert [c.text for c in chunks] == ["Hello", ", world", "!", ""]

    def test_skips_blank_lines_interspersed(self):
        lines_with_blanks = [FIXTURE_LINES[0], "", "  ", FIXTURE_LINES[1]]
        chunks = list(iter_stream_chunks(lines_with_blanks))
        assert len(chunks) == 2


class TestAccumulateStream:
    def _chunks_with_timestamps(self, start: float):
        chunks = list(iter_stream_chunks(FIXTURE_LINES))
        # Simulate chunks arriving at increasing, distinct wall-clock times.
        timestamps = [start + 0.1, start + 0.2, start + 0.3, start + 0.35]
        return list(zip(chunks, timestamps))

    def test_full_text_is_concatenation_of_chunk_texts(self):
        start = 100.0
        result = accumulate_stream(self._chunks_with_timestamps(start), start)
        assert result.full_text == "Hello, world!"

    def test_ttft_is_time_of_first_nonempty_chunk(self):
        start = 100.0
        result = accumulate_stream(self._chunks_with_timestamps(start), start)
        assert result.ttft_seconds == pytest.approx(0.1)

    def test_total_seconds_is_time_of_last_chunk(self):
        start = 100.0
        result = accumulate_stream(self._chunks_with_timestamps(start), start)
        assert result.total_seconds == pytest.approx(0.35)

    def test_tokens_per_second_computed_from_final_done_chunk(self):
        start = 100.0
        result = accumulate_stream(self._chunks_with_timestamps(start), start)
        # eval_count=3, eval_duration=300ms -> 10 tokens/sec
        assert result.tokens_per_second == pytest.approx(10.0)

    def test_chunk_count_matches_number_of_chunks(self):
        start = 100.0
        result = accumulate_stream(self._chunks_with_timestamps(start), start)
        assert result.chunk_count == 4

    def test_empty_stream_returns_all_none_and_empty_text(self):
        result = accumulate_stream([], start_time=100.0)
        assert result.full_text == ""
        assert result.chunk_count == 0
        assert result.ttft_seconds is None
        assert result.total_seconds is None
        assert result.tokens_per_second is None

    def test_ttft_is_none_when_every_chunk_has_empty_text(self):
        start = 100.0
        chunk = StreamChunk(text="", done=True, eval_count=1, eval_duration_ns=100_000_000)
        result = accumulate_stream([(chunk, start + 0.05)], start)
        assert result.ttft_seconds is None

    def test_tokens_per_second_none_when_usage_fields_missing(self):
        start = 100.0
        chunk = StreamChunk(text="hi", done=True, eval_count=None, eval_duration_ns=None)
        result = accumulate_stream([(chunk, start + 0.05)], start)
        assert result.tokens_per_second is None
