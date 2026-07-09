import pytest

from local_ai_agents.tools.file_search import SearchFilesArgs, make_file_search_tool, search_files
from local_ai_agents.tools.sandbox import PathTraversalError


def make_sandbox(tmp_path):
    (tmp_path / "password_reset.md").write_text("Password reset links expire in 15 minutes.")
    (tmp_path / "billing.md").write_text("Billing is charged monthly.")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "notes.md").write_text("General notes about passwords and security.")
    return tmp_path


class TestSearchFiles:
    def test_finds_a_file_by_filename_match(self, tmp_path):
        make_sandbox(tmp_path)
        results = search_files(tmp_path, "password_reset")
        assert "password_reset.md" in results

    def test_finds_a_file_by_content_match(self, tmp_path):
        make_sandbox(tmp_path)
        results = search_files(tmp_path, "charged monthly")
        assert "billing.md" in results

    def test_searches_nested_directories(self, tmp_path):
        make_sandbox(tmp_path)
        results = search_files(tmp_path, "security")
        assert any("notes.md" in r for r in results)

    def test_respects_max_results(self, tmp_path):
        for i in range(5):
            (tmp_path / f"doc{i}.md").write_text("shared content")
        results = search_files(tmp_path, "shared", max_results=2)
        assert len(results) == 2

    def test_no_matches_returns_empty_list(self, tmp_path):
        make_sandbox(tmp_path)
        assert search_files(tmp_path, "nonexistent_query_xyz") == []

    def test_rejects_a_root_path_outside_the_sandbox(self, tmp_path):
        make_sandbox(tmp_path)
        with pytest.raises(PathTraversalError):
            search_files(tmp_path, "password", root_path="../../etc")


class TestMakeFileSearchTool:
    async def test_handler_returns_matches(self, tmp_path):
        make_sandbox(tmp_path)
        tool = make_file_search_tool(tmp_path)
        results = await tool.handler(SearchFilesArgs(query="password_reset"))
        assert "password_reset.md" in results

    def test_tool_is_not_dangerous(self, tmp_path):
        tool = make_file_search_tool(tmp_path)
        assert tool.dangerous is False
