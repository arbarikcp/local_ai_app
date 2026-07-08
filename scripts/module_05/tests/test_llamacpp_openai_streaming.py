import pytest

import llamacpp_openai_streaming as sut


class TestAccumulateChatStream:
    def test_full_text_is_concatenation(self):
        start = 100.0
        chunks = [("Hello", 100.1), (", world", 100.2), ("!", 100.3)]
        result = sut.accumulate_chat_stream(chunks, start)
        assert result.full_text == "Hello, world!"

    def test_ttft_is_time_of_first_nonempty_chunk(self):
        start = 100.0
        # First chunk is an empty-content role-only delta, as OpenAI-style streams often send.
        chunks = [("", 100.05), ("Hello", 100.15)]
        result = sut.accumulate_chat_stream(chunks, start)
        assert result.ttft_seconds == pytest.approx(0.15)

    def test_total_seconds_is_time_of_last_chunk(self):
        start = 100.0
        chunks = [("a", 100.1), ("b", 100.4)]
        result = sut.accumulate_chat_stream(chunks, start)
        assert result.total_seconds == pytest.approx(0.4)

    def test_chunk_count_counts_every_chunk_including_empty_ones(self):
        start = 100.0
        chunks = [("", 100.05), ("Hello", 100.15), ("", 100.2)]
        result = sut.accumulate_chat_stream(chunks, start)
        assert result.chunk_count == 3

    def test_empty_stream_returns_nones(self):
        result = sut.accumulate_chat_stream([], start_time=100.0)
        assert result.full_text == ""
        assert result.chunk_count == 0
        assert result.ttft_seconds is None
        assert result.total_seconds is None


def test_check_openai_client_importable_returns_bool():
    assert isinstance(sut.check_openai_client_importable(), bool)


def test_main_skips_cleanly_when_server_unreachable(capsys):
    exit_code = sut.main(["--base-url", "http://localhost:8080/v1", "--prompt", "hi"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "SKIPPED" in (captured.out + captured.err)
