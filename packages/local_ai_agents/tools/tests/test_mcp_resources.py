import pytest

from local_ai_agents.tools.mcp_resources import ResourceNotFoundError, ResourceRegistry
from local_ai_agents.tools.sandbox import PathTraversalError


class TestRegister:
    def test_a_valid_relative_uri_registers_successfully(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        registry = ResourceRegistry(tmp_path)
        registry.register("notes.txt", "some notes")
        assert len(registry.list()) == 1

    def test_a_uri_outside_the_sandbox_is_rejected_at_registration_time(self, tmp_path):
        registry = ResourceRegistry(tmp_path)
        with pytest.raises(PathTraversalError):
            registry.register("../../etc/passwd", "malicious")


class TestList:
    def test_lists_every_registered_resource(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        registry = ResourceRegistry(tmp_path)
        registry.register("a.txt", "resource a")
        registry.register("b.txt", "resource b")
        assert {d.uri for d in registry.list()} == {"a.txt", "b.txt"}

    def test_empty_registry_lists_nothing(self, tmp_path):
        assert ResourceRegistry(tmp_path).list() == []


class TestRead:
    def test_returns_the_real_file_content(self, tmp_path):
        (tmp_path / "notes.txt").write_text("the actual content")
        registry = ResourceRegistry(tmp_path)
        registry.register("notes.txt", "notes")
        content = registry.read("notes.txt")
        assert content.text == "the actual content"

    def test_raises_for_an_unregistered_uri(self, tmp_path):
        registry = ResourceRegistry(tmp_path)
        with pytest.raises(ResourceNotFoundError):
            registry.read("never_registered.txt")
