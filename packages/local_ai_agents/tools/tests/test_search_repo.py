import pytest

from local_ai_agents.tools.sandbox import PathTraversalError
from local_ai_agents.tools.search_repo import SearchRepoArgs, make_search_repo_tool, search_repo


def make_sample_repo(tmp_path):
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")
    (tmp_path / "notes.txt").write_text("add this to notes")  # not a .py file, should be ignored
    return tmp_path


class TestSearchRepo:
    def test_finds_a_match_with_its_line_number(self, tmp_path):
        make_sample_repo(tmp_path)
        matches = search_repo(tmp_path, "return a + b")
        assert matches[0].path == "calculator.py"
        assert matches[0].line_number == 2

    def test_only_searches_python_files(self, tmp_path):
        make_sample_repo(tmp_path)
        matches = search_repo(tmp_path, "add")
        assert all(m.path.endswith(".py") for m in matches)

    def test_respects_max_results(self, tmp_path):
        for i in range(5):
            (tmp_path / f"mod{i}.py").write_text("shared_marker = 1\n")
        matches = search_repo(tmp_path, "shared_marker", max_results=2)
        assert len(matches) == 2

    def test_no_matches_returns_empty_list(self, tmp_path):
        make_sample_repo(tmp_path)
        assert search_repo(tmp_path, "nonexistent_xyz") == []

    def test_rejects_a_root_path_outside_the_sandbox(self, tmp_path):
        make_sample_repo(tmp_path)
        with pytest.raises(PathTraversalError):
            search_repo(tmp_path, "add", root_path="../../etc")


class TestMakeSearchRepoTool:
    async def test_handler_returns_serializable_dicts(self, tmp_path):
        make_sample_repo(tmp_path)
        tool = make_search_repo_tool(tmp_path)
        result = await tool.handler(SearchRepoArgs(query="add"))
        assert isinstance(result[0], dict)
        assert "line_number" in result[0]
