import pytest

from local_ai_agents.planners.memory import AgentMemory
from local_ai_agents.planners.workflow_graph import END, GraphConfigurationError, WorkflowGraph, WorkflowNode


async def noop_node(state: dict, memory: AgentMemory) -> dict:
    return state


def make_linear_graph() -> WorkflowGraph:
    graph = WorkflowGraph(start_node="classify")
    graph.add_node(WorkflowNode(name="classify", run=noop_node))
    graph.add_node(WorkflowNode(name="gather", run=noop_node))
    graph.add_node(WorkflowNode(name="done", run=noop_node))
    graph.add_edge("classify", "gather")
    graph.add_edge("gather", "done")
    graph.add_edge("done", END)
    return graph


class TestLinearTraversal:
    def test_follows_the_single_edge_from_each_node(self):
        graph = make_linear_graph()
        assert graph.next_node("classify", {}) == "gather"
        assert graph.next_node("gather", {}) == "done"
        assert graph.next_node("done", {}) == END

    def test_a_node_with_no_outgoing_edges_returns_none(self):
        graph = WorkflowGraph(start_node="a")
        graph.add_node(WorkflowNode(name="a", run=noop_node))
        assert graph.next_node("a", {}) is None


class TestBranching:
    def test_the_first_matching_condition_wins(self):
        graph = WorkflowGraph(start_node="check")
        graph.add_node(WorkflowNode(name="check", run=noop_node))
        graph.add_node(WorkflowNode(name="approved_path", run=noop_node))
        graph.add_node(WorkflowNode(name="denied_path", run=noop_node))
        graph.add_edge("check", "approved_path", condition=lambda s: s.get("approved") is True)
        graph.add_edge("check", "denied_path", condition=lambda s: True)  # fallback

        assert graph.next_node("check", {"approved": True}) == "approved_path"
        assert graph.next_node("check", {"approved": False}) == "denied_path"

    def test_no_matching_condition_returns_none(self):
        graph = WorkflowGraph(start_node="check")
        graph.add_node(WorkflowNode(name="check", run=noop_node))
        graph.add_node(WorkflowNode(name="only_path", run=noop_node))
        graph.add_edge("check", "only_path", condition=lambda s: s.get("ready") is True)
        assert graph.next_node("check", {"ready": False}) is None


class TestConfigurationErrors:
    def test_rejects_an_edge_from_an_unregistered_node(self):
        graph = WorkflowGraph(start_node="a")
        graph.add_node(WorkflowNode(name="a", run=noop_node))
        with pytest.raises(GraphConfigurationError):
            graph.add_edge("missing", "a")

    def test_rejects_an_edge_to_an_unregistered_node(self):
        graph = WorkflowGraph(start_node="a")
        graph.add_node(WorkflowNode(name="a", run=noop_node))
        with pytest.raises(GraphConfigurationError):
            graph.add_edge("a", "missing")

    def test_an_edge_to_end_is_always_allowed(self):
        graph = WorkflowGraph(start_node="a")
        graph.add_node(WorkflowNode(name="a", run=noop_node))
        graph.add_edge("a", END)  # should not raise


class TestNodeNames:
    def test_lists_every_registered_node(self):
        graph = make_linear_graph()
        assert set(graph.node_names()) == {"classify", "gather", "done"}
