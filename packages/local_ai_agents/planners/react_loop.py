"""ReActLoop (theory doc §3, Lab 1) - alternates: ask the LLM to reason
about what to do next and propose either a tool call or a final answer;
execute the tool via Module 14's `ToolExecutor`; append the observation;
repeat. Stops on a final answer, a safety-budget limit, or loop-prevention
tripping. This is the "avoid" shape from the preferred mental model,
deliberately implemented so Lab 2 can break it with adversarial prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest
from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.planners.loop_prevention import LoopDetectedError, LoopGuard
from local_ai_agents.planners.memory import AgentMemory
from local_ai_agents.planners.safety_budget import AgentSafetyBudget, SafetyBudgetExceededError
from local_ai_agents.tools.base import ToolCallProposal
from local_ai_agents.tools.registry import ToolRegistry

REACT_PROMPT_TEMPLATE = """You are solving a task step by step, using tools when needed.

Available tools:
{tool_schemas}

Task:
{request}

History so far:
{transcript}

Decide your next step. Respond with a JSON object with exactly two fields: "action" \
("tool_call" or "final_answer") and either "tool"/"arguments" (for tool_call) or "answer" \
(for final_answer). Respond with only the JSON object, nothing else."""


class ReActParseError(Exception):
    pass


@dataclass(frozen=True)
class ReActResult:
    final_answer: str | None
    stopped_reason: str  # "final_answer" | "safety_budget" | "loop_detected"
    memory: AgentMemory


class ReActLoop:
    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        runtime: LLMRuntime,
        model: str,
        *,
        loop_guard: LoopGuard | None = None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._runtime = runtime
        self._model = model
        self._loop_guard = loop_guard or LoopGuard()

    async def run(self, request: str, safety_budget: AgentSafetyBudget, *, role: str = "default") -> ReActResult:
        memory = AgentMemory()
        tool_schemas = json.dumps(self._registry.schema_list(), indent=2)

        while True:
            try:
                safety_budget.record_step()
                safety_budget.check_runtime()
            except SafetyBudgetExceededError:
                return ReActResult(final_answer=None, stopped_reason="safety_budget", memory=memory)

            prompt = REACT_PROMPT_TEMPLATE.format(
                tool_schemas=tool_schemas, request=request, transcript=memory.transcript()
            )
            response = await self._runtime.generate(LLMRequest(model=self._model, prompt=prompt))
            safety_budget.record_tokens((response.prompt_tokens or 0) + (response.completion_tokens or 0))

            try:
                parsed = json.loads(response.text.strip())
            except json.JSONDecodeError as exc:
                raise ReActParseError(f"Could not parse ReAct step: {response.text!r}") from exc

            action = parsed.get("action")

            if action == "final_answer":
                answer = parsed.get("answer", "")
                memory.add("reasoning", f"Final answer: {answer}")
                return ReActResult(final_answer=answer, stopped_reason="final_answer", memory=memory)

            if action == "tool_call":
                tool_name = parsed["tool"]
                arguments = parsed.get("arguments", {})
                memory.add("tool_call", f"{tool_name}({arguments})", data={"tool": tool_name, "arguments": arguments})

                try:
                    self._loop_guard.record(tool_name, arguments)
                except LoopDetectedError:
                    return ReActResult(final_answer=None, stopped_reason="loop_detected", memory=memory)

                try:
                    safety_budget.record_tool_call()
                except SafetyBudgetExceededError:
                    return ReActResult(final_answer=None, stopped_reason="safety_budget", memory=memory)

                result = await self._executor.execute(
                    ToolCallProposal(tool_name=tool_name, raw_arguments=arguments), role=role
                )
                memory.add("observation", result.as_text(), data={"success": result.success})
                continue

            raise ReActParseError(f"Unknown action {action!r} in ReAct step: {response.text!r}")
