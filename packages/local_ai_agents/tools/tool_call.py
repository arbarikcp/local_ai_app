"""Tool selection (theory doc §4) - one LLM call, given the registry's
tool schemas and a user request, asked to respond with a JSON tool-call
proposal. Parsed strictly: a real `ToolCallParseError` on malformed JSON
or a missing `tool` field, not a silent fallback - the same honesty
standard Module 8's structured-output reliability ladder established.
"""

from __future__ import annotations

import json

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest
from local_ai_agents.tools.base import ToolCallProposal
from local_ai_agents.tools.registry import ToolRegistry

TOOL_CALL_PROMPT_TEMPLATE = """You have access to the following tools:

{tool_schemas}

Given the user's request, decide which tool (if any) to call. Respond with a JSON object with \
exactly two fields: "tool" (the tool name, or null if no tool applies) and "arguments" (an \
object of argument name to value, or {{}} if none). Respond with only the JSON object, nothing \
else.

User request:
{request}"""


class ToolCallParseError(Exception):
    """A real failure mode for small local models proposing tool calls -
    not swallowed silently, same discipline as Module 13's `JudgeParseError`.
    """


def build_tool_call_prompt(request: str, registry: ToolRegistry) -> str:
    tool_schemas = json.dumps(registry.schema_list(), indent=2)
    return TOOL_CALL_PROMPT_TEMPLATE.format(tool_schemas=tool_schemas, request=request)


async def propose_tool_call(request: str, registry: ToolRegistry, runtime: LLMRuntime, model: str) -> ToolCallProposal | None:
    """Returns `None` if the model decided no tool applies (`"tool": null`)
    - a legitimate outcome, not a parse failure.
    """
    prompt = build_tool_call_prompt(request, registry)
    response = await runtime.generate(LLMRequest(model=model, prompt=prompt))
    try:
        parsed = json.loads(response.text.strip())
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"Could not parse tool-call response as JSON: {response.text!r}") from exc

    if "tool" not in parsed:
        raise ToolCallParseError(f"Tool-call response missing 'tool' field: {response.text!r}")

    tool_name = parsed["tool"]
    if tool_name is None:
        return None
    return ToolCallProposal(tool_name=tool_name, raw_arguments=parsed.get("arguments", {}))
