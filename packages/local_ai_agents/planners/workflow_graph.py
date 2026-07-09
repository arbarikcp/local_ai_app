"""WorkflowGraph (theory doc §4-5) - one engine for both "state machine
agents" and "graph-based agents": a state machine is a graph whose
conditional edges each have exactly one true branch. Nodes are either
deterministic functions or a single bounded LLM decision point;
transitions between nodes are conditional functions evaluated against the
workflow's current state, not the model's free choice of what to do next
- the entire point of the "prefer" shape in the theory doc's mental model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from local_ai_agents.planners.memory import AgentMemory

WorkflowState = dict[str, Any]
NodeFn = Callable[[WorkflowState, AgentMemory], Awaitable[WorkflowState]]
ConditionFn = Callable[[WorkflowState], bool]

END = "__end__"


class GraphConfigurationError(Exception):
    pass


def _always(state: WorkflowState) -> bool:
    return True


@dataclass(frozen=True)
class WorkflowNode:
    name: str
    run: NodeFn
    requires_approval_tool: str | None = None


@dataclass(frozen=True)
class Edge:
    from_node: str
    to_node: str
    condition: ConditionFn = field(default=_always)


class WorkflowGraph:
    def __init__(self, start_node: str) -> None:
        self.start_node = start_node
        self._nodes: dict[str, WorkflowNode] = {}
        self._edges: dict[str, list[Edge]] = {}

    def add_node(self, node: WorkflowNode) -> None:
        self._nodes[node.name] = node
        self._edges.setdefault(node.name, [])

    def add_edge(self, from_node: str, to_node: str, condition: ConditionFn = _always) -> None:
        if from_node not in self._nodes:
            raise GraphConfigurationError(f"Unknown from_node '{from_node}'")
        if to_node not in self._nodes and to_node != END:
            raise GraphConfigurationError(f"Unknown to_node '{to_node}'")
        self._edges.setdefault(from_node, []).append(Edge(from_node=from_node, to_node=to_node, condition=condition))

    def get_node(self, name: str) -> WorkflowNode:
        return self._nodes[name]

    def next_node(self, current: str, state: WorkflowState) -> str | None:
        """First edge whose condition is true, in registration order -
        deterministic, not the model's free choice. Returns None if no
        edge matches - a dead end (misconfigured graph, or a node that
        should have an unconditional edge to `END` but doesn't).
        """
        for edge in self._edges.get(current, []):
            if edge.condition(state):
                return edge.to_node
        return None

    def node_names(self) -> list[str]:
        return list(self._nodes.keys())
