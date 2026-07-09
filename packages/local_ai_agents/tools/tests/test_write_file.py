import pytest

from local_ai_agents.tools.sandbox import PathTraversalError
from local_ai_agents.tools.write_file import WriteFileArgs, make_write_file_tool, write_file


class TestWriteFile:
    def test_writes_content_to_the_target_path(self, tmp_path):
        write_file(tmp_path, "notes.txt", "hello world")
        assert (tmp_path / "notes.txt").read_text() == "hello world"

    def test_creates_parent_directories_as_needed(self, tmp_path):
        write_file(tmp_path, "sub/dir/notes.txt", "nested")
        assert (tmp_path / "sub" / "dir" / "notes.txt").read_text() == "nested"

    def test_returns_the_path_relative_to_the_sandbox(self, tmp_path):
        result = write_file(tmp_path, "notes.txt", "hello")
        assert result == "notes.txt"

    def test_rejects_a_path_outside_the_sandbox(self, tmp_path):
        with pytest.raises(PathTraversalError):
            write_file(tmp_path, "../../etc/passwd", "malicious")

    def test_rejects_an_absolute_path_override(self, tmp_path):
        with pytest.raises(PathTraversalError):
            write_file(tmp_path, "/etc/passwd", "malicious")


class TestMakeWriteFileTool:
    def test_tool_is_dangerous(self, tmp_path):
        tool = make_write_file_tool(tmp_path)
        assert tool.dangerous is True

    async def test_handler_writes_the_file(self, tmp_path):
        tool = make_write_file_tool(tmp_path)
        await tool.handler(WriteFileArgs(path="notes.txt", content="hi"))
        assert (tmp_path / "notes.txt").read_text() == "hi"
