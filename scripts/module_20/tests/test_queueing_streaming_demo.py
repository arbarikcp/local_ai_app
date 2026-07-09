import queueing_streaming_demo as sut


class TestRunLab:
    async def test_every_request_is_served(self):
        result = await sut.run_lab()
        assert result["request_count"] == 4
        assert len(result["streamed_chunk_counts"]) == 4

    async def test_every_request_produces_real_streamed_chunks(self):
        result = await sut.run_lab()
        assert all(count > 0 for count in result["streamed_chunk_counts"])

    async def test_queue_drains_completely(self):
        result = await sut.run_lab()
        assert result["final_running_count"] == 0
        assert result["final_waiting_count"] == 0


class TestResultToMarkdown:
    async def test_markdown_mentions_the_request_count(self):
        result = await sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "4" in markdown
