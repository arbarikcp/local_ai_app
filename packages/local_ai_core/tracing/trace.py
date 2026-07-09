"""TraceBuilder — a real span tree matching curriculum's trace model
exactly (theory doc §3, §8-10). Built with a context-manager-style
`span()` so every span's `elapsed_ms` is real measured time (Module 6's
`Timer`, reused) around whatever work actually happens inside the `with`
block - not a hand-asserted number.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from local_ai_core.runtimes.base import Timer

# Curriculum's exact trace-model step order (theory doc "Trace model"),
# snake_cased into span names.
REQUEST_TRACE_STEPS = [
    "input_validation",
    "prompt_template_version",
    "retrieval_query",
    "retrieved_chunk_ids",
    "reranker_scores",
    "context_packing",
    "model_call",
    "output_validation",
    "tool_calls",
    "final_response",
    "evaluation_hooks",
]

# The subset every request trace must contain regardless of whether
# retrieval/tool calls happened for this particular request (theory doc:
# "tool calls if any" - optional; the rest are not).
CORE_REQUIRED_STEPS = [
    "input_validation",
    "model_call",
    "final_response",
]


@dataclass(frozen=True)
class TraceSpan:
    name: str
    elapsed_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    request_id: str
    spans: list[TraceSpan] = field(default_factory=list)

    def span_names(self) -> list[str]:
        return [s.name for s in self.spans]

    def total_elapsed_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.spans)


class TraceBuilder:
    def __init__(self, request_id: str) -> None:
        self.trace = Trace(request_id=request_id)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        timer = Timer()
        try:
            yield
        finally:
            self.trace.spans.append(TraceSpan(name=name, elapsed_ms=timer.elapsed_ms, attributes=attributes))

    def record_retrieval_step(
        self, *, query: str, chunk_ids: list[str], reranker_scores: list[float] | None = None
    ) -> None:
        with self.span("retrieval_query", query=query):
            pass
        with self.span("retrieved_chunk_ids", chunk_ids=chunk_ids):
            pass
        if reranker_scores is not None:
            with self.span("reranker_scores", scores=reranker_scores):
                pass

    def record_tool_call_step(
        self, *, tool_name: str, arguments: dict[str, Any], result: Any = None, error: str | None = None
    ) -> None:
        with self.span("tool_calls", tool_name=tool_name, arguments=arguments, result=result, error=error):
            pass

    def record_agent_step(self, *, step_index: int, action: str, observation: str) -> None:
        with self.span("agent_step", step_index=step_index, action=action, observation=observation):
            pass

    def build(self) -> Trace:
        return self.trace


def validate_trace_shape(trace: Trace, required_steps: list[str] = CORE_REQUIRED_STEPS) -> list[str]:
    """Returns the required step names missing from `trace` - empty when
    the trace is complete. Doesn't enforce curriculum's full step order,
    only presence of the always-required core (retrieval/tool-call steps
    are legitimately absent from a request that didn't need them).
    """
    present = set(trace.span_names())
    return [step for step in required_steps if step not in present]
