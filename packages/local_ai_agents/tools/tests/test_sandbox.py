import pytest

from local_ai_agents.tools.sandbox import PathTraversalError, resolve_within_sandbox


class TestResolveWithinSandbox:
    def test_a_plain_relative_path_resolves_inside_the_sandbox(self, tmp_path):
        result = resolve_within_sandbox(tmp_path, "notes.txt")
        assert result == (tmp_path / "notes.txt").resolve()

    def test_a_nested_relative_path_resolves_inside_the_sandbox(self, tmp_path):
        result = resolve_within_sandbox(tmp_path, "subdir/notes.txt")
        assert result == (tmp_path / "subdir" / "notes.txt").resolve()

    def test_dot_root_resolves_to_the_sandbox_itself(self, tmp_path):
        assert resolve_within_sandbox(tmp_path, ".") == tmp_path.resolve()

    def test_parent_traversal_is_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_within_sandbox(tmp_path, "../../etc/passwd")

    def test_absolute_path_override_is_rejected(self, tmp_path):
        # Path("/base") / "/etc/passwd" silently discards the base entirely
        # under plain pathlib semantics - this must still be caught.
        with pytest.raises(PathTraversalError):
            resolve_within_sandbox(tmp_path, "/etc/passwd")

    def test_traversal_that_returns_back_inside_the_sandbox_is_allowed(self, tmp_path):
        (tmp_path / "a").mkdir()
        result = resolve_within_sandbox(tmp_path, "a/../notes.txt")
        assert result == (tmp_path / "notes.txt").resolve()
