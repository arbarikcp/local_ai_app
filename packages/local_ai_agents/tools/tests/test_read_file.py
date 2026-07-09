import pytest

from local_ai_agents.tools.read_file import ReadFileArgs, make_read_file_tool, read_file_lines


def make_sample(tmp_path):
    (tmp_path / "sample.py").write_text("line1\nline2\nline3\nline4\nline5\n")
    return tmp_path


class TestReadFileLines:
    def test_reads_the_whole_file_by_default(self, tmp_path):
        make_sample(tmp_path)
        assert read_file_lines(tmp_path, "sample.py") == "line1\nline2\nline3\nline4\nline5"

    def test_reads_a_specific_line_range(self, tmp_path):
        make_sample(tmp_path)
        assert read_file_lines(tmp_path, "sample.py", start_line=2, end_line=3) == "line2\nline3"

    def test_reads_from_a_start_line_to_the_end(self, tmp_path):
        make_sample(tmp_path)
        assert read_file_lines(tmp_path, "sample.py", start_line=4) == "line4\nline5"

    def test_rejects_start_line_after_end_line(self, tmp_path):
        make_sample(tmp_path)
        with pytest.raises(ValueError):
            read_file_lines(tmp_path, "sample.py", start_line=5, end_line=2)


class TestMakeReadFileTool:
    async def test_handler_returns_the_requested_range(self, tmp_path):
        make_sample(tmp_path)
        tool = make_read_file_tool(tmp_path)
        result = await tool.handler(ReadFileArgs(path="sample.py", start_line=1, end_line=2))
        assert result == "line1\nline2"
