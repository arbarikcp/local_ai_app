from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool, ToolCallProposal, ToolResult


class DummyArgs(BaseModel):
    value: int = Field(ge=0, le=100)


async def dummy_handler(args: DummyArgs) -> int:
    return args.value * 2


class TestToolResult:
    def test_success_result_text_is_the_data(self):
        result = ToolResult(success=True, data="42")
        assert result.as_text() == "42"

    def test_failure_result_text_includes_the_error_message(self):
        result = ToolResult(success=False, error_message="bad input")
        assert result.as_text() == "Error: bad input"

    def test_non_string_data_is_stringified(self):
        result = ToolResult(success=True, data=42)
        assert result.as_text() == "42"


class TestTool:
    def test_json_schema_includes_name_and_dangerous_flag(self):
        tool = Tool(name="dummy", description="A dummy tool.", args_model=DummyArgs, handler=dummy_handler)
        schema = tool.json_schema()
        assert schema["name"] == "dummy"
        assert schema["dangerous"] is False

    def test_json_schema_includes_the_pydantic_parameters_schema(self):
        tool = Tool(name="dummy", description="A dummy tool.", args_model=DummyArgs, handler=dummy_handler)
        schema = tool.json_schema()
        assert "value" in schema["parameters"]["properties"]

    def test_dangerous_flag_defaults_to_false(self):
        tool = Tool(name="dummy", description="d", args_model=DummyArgs, handler=dummy_handler)
        assert tool.dangerous is False

    def test_dangerous_flag_can_be_set(self):
        tool = Tool(name="dummy", description="d", args_model=DummyArgs, handler=dummy_handler, dangerous=True)
        assert tool.dangerous is True


class TestToolCallProposal:
    def test_defaults_to_empty_arguments(self):
        proposal = ToolCallProposal(tool_name="dummy")
        assert proposal.raw_arguments == {}

    def test_carries_raw_arguments(self):
        proposal = ToolCallProposal(tool_name="dummy", raw_arguments={"value": 5})
        assert proposal.raw_arguments == {"value": 5}
