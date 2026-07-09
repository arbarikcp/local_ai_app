import pytest
from pydantic import BaseModel

from local_ai_agents.tools.base import Tool, ToolNotFoundError
from local_ai_agents.tools.registry import ToolRegistry


class DummyArgs(BaseModel):
    value: int


async def dummy_handler(args: DummyArgs) -> int:
    return args.value


def make_tool(name: str = "dummy") -> Tool:
    return Tool(name=name, description="d", args_model=DummyArgs, handler=dummy_handler)


class TestRegister:
    def test_registered_tool_is_retrievable_by_name(self):
        registry = ToolRegistry()
        registry.register(make_tool("dummy"))
        assert registry.get("dummy").name == "dummy"

    def test_registering_the_same_name_twice_overwrites(self):
        registry = ToolRegistry()
        registry.register(make_tool("dummy"))
        registry.register(make_tool("dummy"))
        assert len(registry) == 1

    def test_contains_reflects_registered_names(self):
        registry = ToolRegistry()
        registry.register(make_tool("dummy"))
        assert "dummy" in registry
        assert "missing" not in registry


class TestGet:
    def test_raises_for_an_unregistered_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("missing")


class TestListTools:
    def test_lists_every_registered_tool(self):
        registry = ToolRegistry()
        registry.register(make_tool("a"))
        registry.register(make_tool("b"))
        assert {t.name for t in registry.list_tools()} == {"a", "b"}

    def test_empty_registry_lists_nothing(self):
        assert ToolRegistry().list_tools() == []


class TestSchemaList:
    def test_returns_one_schema_per_tool(self):
        registry = ToolRegistry()
        registry.register(make_tool("a"))
        registry.register(make_tool("b"))
        schemas = registry.schema_list()
        assert len(schemas) == 2
        assert {s["name"] for s in schemas} == {"a", "b"}
