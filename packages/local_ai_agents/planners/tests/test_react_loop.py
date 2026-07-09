from pydantic import BaseModel

from local_ai_core.runtimes.types import LLMRequest, LLMResponse
from local_ai_agents.executors.tool_executor import ToolExecutor
from local_ai_agents.planners.loop_prevention import LoopGuard
from local_ai_agents.planners.react_loop import ReActLoop
from local_ai_agents.planners.safety_budget import AgentSafetyBudget
from local_ai_agents.tools.base import Tool
from local_ai_agents.tools.registry import ToolRegistry


class AddArgs(BaseModel):
    a: int
    b: int


async def add_handler(args: AddArgs) -> int:
    return args.a + args.b


class SequencedRuntime:
    """Returns one scripted response per call, in order - a real,
    deterministic multi-turn stand-in `FakeRuntime` can't provide (it
    returns the same text every call for a given model).
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        text = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return LLMResponse(text=text, model=request.model, prompt_tokens=10, completion_tokens=10, latency_ms=0.0, stop_reason="stop")

    async def stream(self, request: LLMRequest):  # pragma: no cover
        raise NotImplementedError

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        return [0] * len(rendered_prompt.split())


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="add", description="add two numbers", args_model=AddArgs, handler=add_handler))
    return registry


def make_budget(**overrides) -> AgentSafetyBudget:
    defaults = dict(max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000)
    defaults.update(overrides)
    return AgentSafetyBudget(**defaults)


class TestReActLoop:
    async def test_immediate_final_answer(self):
        runtime = SequencedRuntime(['{"action": "final_answer", "answer": "42"}'])
        loop = ReActLoop(make_registry(), ToolExecutor(make_registry()), runtime, model="fake-model")
        result = await loop.run("what is the answer?", make_budget())
        assert result.stopped_reason == "final_answer"
        assert result.final_answer == "42"

    async def test_a_tool_call_then_a_final_answer(self):
        runtime = SequencedRuntime(
            [
                '{"action": "tool_call", "tool": "add", "arguments": {"a": 2, "b": 3}}',
                '{"action": "final_answer", "answer": "5"}',
            ]
        )
        registry = make_registry()
        loop = ReActLoop(registry, ToolExecutor(registry), runtime, model="fake-model")
        result = await loop.run("what is 2 + 3?", make_budget())
        assert result.final_answer == "5"
        assert len(result.memory) == 3  # tool_call, observation, reasoning(final)

    async def test_observation_is_recorded_in_memory(self):
        runtime = SequencedRuntime(
            [
                '{"action": "tool_call", "tool": "add", "arguments": {"a": 2, "b": 3}}',
                '{"action": "final_answer", "answer": "5"}',
            ]
        )
        registry = make_registry()
        loop = ReActLoop(registry, ToolExecutor(registry), runtime, model="fake-model")
        result = await loop.run("what is 2 + 3?", make_budget())
        observations = [e for e in result.memory.entries() if e.kind == "observation"]
        assert observations[0].content == "5"

    async def test_stops_when_the_safety_budget_is_exhausted(self):
        # Always proposes a *different* tool call so LoopGuard never trips,
        # isolating the safety-budget stop path specifically.
        responses = [f'{{"action": "tool_call", "tool": "add", "arguments": {{"a": 1, "b": {i}}}}}' for i in range(10)]
        runtime = SequencedRuntime(responses)
        registry = make_registry()
        loop = ReActLoop(registry, ToolExecutor(registry), runtime, model="fake-model", loop_guard=LoopGuard(max_repeats=100))
        result = await loop.run("keep going", make_budget(max_steps=3))
        assert result.stopped_reason == "safety_budget"
        assert result.final_answer is None

    async def test_stops_when_loop_guard_trips_on_a_repeated_call(self):
        runtime = SequencedRuntime(['{"action": "tool_call", "tool": "add", "arguments": {"a": 1, "b": 1}}'])
        registry = make_registry()
        loop = ReActLoop(registry, ToolExecutor(registry), runtime, model="fake-model", loop_guard=LoopGuard(max_repeats=3))
        result = await loop.run("keep adding 1+1", make_budget(max_steps=20))
        assert result.stopped_reason == "loop_detected"
