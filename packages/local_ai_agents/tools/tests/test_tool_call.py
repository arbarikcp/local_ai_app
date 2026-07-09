import pytest
from pydantic import BaseModel

from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.registry import ToolRegistry
from local_ai_agents.tools.tool_call import ToolCallParseError, build_tool_call_prompt, propose_tool_call


class DummyArgs(BaseModel):
    value: int


async def dummy_handler(args: DummyArgs) -> int:
    return args.value


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="dummy", description="A dummy tool.", args_model=DummyArgs, handler=dummy_handler))
    return registry


class TestBuildToolCallPrompt:
    def test_includes_the_request_and_tool_schemas(self):
        prompt = build_tool_call_prompt("do the thing", make_registry())
        assert "do the thing" in prompt
        assert "dummy" in prompt


class TestProposeToolCall:
    async def test_parses_a_valid_proposal(self):
        runtime = FakeRuntime(default_response='{"tool": "dummy", "arguments": {"value": 5}}')
        proposal = await propose_tool_call("do it", make_registry(), runtime, model="fake-model")
        assert proposal is not None
        assert proposal.tool_name == "dummy"
        assert proposal.raw_arguments == {"value": 5}

    async def test_returns_none_when_the_model_declines_to_call_a_tool(self):
        runtime = FakeRuntime(default_response='{"tool": null, "arguments": {}}')
        proposal = await propose_tool_call("just chatting", make_registry(), runtime, model="fake-model")
        assert proposal is None

    async def test_raises_on_unparseable_response(self):
        runtime = FakeRuntime(default_response="not json")
        with pytest.raises(ToolCallParseError):
            await propose_tool_call("do it", make_registry(), runtime, model="fake-model")

    async def test_raises_when_tool_field_is_missing(self):
        runtime = FakeRuntime(default_response='{"arguments": {}}')
        with pytest.raises(ToolCallParseError):
            await propose_tool_call("do it", make_registry(), runtime, model="fake-model")

    async def test_defaults_to_empty_arguments_when_omitted(self):
        runtime = FakeRuntime(default_response='{"tool": "dummy"}')
        proposal = await propose_tool_call("do it", make_registry(), runtime, model="fake-model")
        assert proposal.raw_arguments == {}
