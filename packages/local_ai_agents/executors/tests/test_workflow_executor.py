from local_ai_agents.executors.workflow_executor import WorkflowExecutor
from local_ai_agents.planners.checkpoint_store import CheckpointStore
from local_ai_agents.planners.memory import AgentMemory
from local_ai_agents.planners.safety_budget import AgentSafetyBudget
from local_ai_agents.planners.workflow_graph import END, WorkflowGraph, WorkflowNode
from local_ai_agents.policies.approval import AutoApprovalGate, NullApprovalGate


def make_budget(**overrides) -> AgentSafetyBudget:
    defaults = dict(max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000)
    defaults.update(overrides)
    return AgentSafetyBudget(**defaults)


async def increment_node(state: dict, memory: AgentMemory) -> dict:
    return {**state, "count": state.get("count", 0) + 1}


async def always_fails_node(state: dict, memory: AgentMemory) -> dict:
    raise RuntimeError("node exploded")


def make_linear_graph() -> WorkflowGraph:
    graph = WorkflowGraph(start_node="step1")
    graph.add_node(WorkflowNode(name="step1", run=increment_node))
    graph.add_node(WorkflowNode(name="step2", run=increment_node))
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)
    return graph


class TestHappyPath:
    async def test_reaches_end_and_accumulates_state(self):
        executor = WorkflowExecutor(make_linear_graph())
        result = await executor.run({"count": 0}, make_budget())
        assert result.stopped_reason == "end"
        assert result.final_state["count"] == 2

    async def test_final_node_is_the_graph_end_sentinel(self):
        executor = WorkflowExecutor(make_linear_graph())
        result = await executor.run({}, make_budget())
        assert result.final_node == END


class TestSafetyBudget:
    async def test_stops_when_max_steps_is_exhausted(self):
        # max_steps=1 lets step1 run, then the budget check fires in front
        # of step2 - final_node reports where it stopped (step2), not the
        # last node that actually completed (step1).
        executor = WorkflowExecutor(make_linear_graph())
        result = await executor.run({}, make_budget(max_steps=1))
        assert result.stopped_reason == "safety_budget"
        assert result.final_node == "step2"


class TestApproval:
    async def test_a_node_requiring_approval_is_denied_by_default(self):
        graph = WorkflowGraph(start_node="write")
        graph.add_node(WorkflowNode(name="write", run=increment_node, requires_approval_tool="file_write"))
        graph.add_edge("write", END)
        executor = WorkflowExecutor(graph, approval_gate=NullApprovalGate())
        result = await executor.run({}, make_budget())
        assert result.stopped_reason == "approval_denied"

    async def test_a_node_requiring_approval_proceeds_when_approved(self):
        graph = WorkflowGraph(start_node="write")
        graph.add_node(WorkflowNode(name="write", run=increment_node, requires_approval_tool="file_write"))
        graph.add_edge("write", END)
        executor = WorkflowExecutor(graph, approval_gate=AutoApprovalGate())
        result = await executor.run({"count": 0}, make_budget())
        assert result.stopped_reason == "end"
        assert result.final_state["count"] == 1


class TestFailureRecovery:
    async def test_a_node_that_always_fails_stops_with_failed_after_retries(self):
        graph = WorkflowGraph(start_node="broken")
        graph.add_node(WorkflowNode(name="broken", run=always_fails_node))
        graph.add_edge("broken", END)
        executor = WorkflowExecutor(graph, max_retries=2)
        result = await executor.run({}, make_budget())
        assert result.stopped_reason == "failed"

    async def test_retries_are_recorded_in_memory(self):
        graph = WorkflowGraph(start_node="broken")
        graph.add_node(WorkflowNode(name="broken", run=always_fails_node))
        graph.add_edge("broken", END)
        executor = WorkflowExecutor(graph, max_retries=2)
        result = await executor.run({}, make_budget())
        # "attempt " (trailing space) matches the 3 per-attempt failure lines
        # ("attempt 1 failed", "attempt 2 failed", "attempt 3 failed") but not
        # the final summary line ("failed after 3 attempts:").
        attempt_entries = [e for e in result.memory.entries() if "attempt " in e.content]
        assert len(attempt_entries) == 3  # initial try + 2 retries


class TestMisconfiguredGraph:
    async def test_a_node_with_no_matching_outgoing_edge_fails(self):
        graph = WorkflowGraph(start_node="dead_end")
        graph.add_node(WorkflowNode(name="dead_end", run=increment_node))
        executor = WorkflowExecutor(graph)
        result = await executor.run({}, make_budget())
        assert result.stopped_reason == "failed"


class TestCheckpointingAndResume:
    async def test_a_checkpoint_is_saved_after_every_successful_step(self, tmp_path):
        store = CheckpointStore(tmp_path / "checkpoints.db")
        executor = WorkflowExecutor(make_linear_graph(), checkpoint_store=store)
        await executor.run({"count": 0}, make_budget(), run_id="run-1")

        checkpoint = store.load("run-1")
        assert checkpoint is not None
        assert checkpoint.node_name == END  # the run completed, so the resume point is END
        store.close()

    async def test_resuming_continues_from_the_last_checkpointed_node(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"

        # First executor: a graph where step2 always fails, so the run stops
        # after step1's checkpoint is saved but before step2 completes.
        graph = WorkflowGraph(start_node="step1")
        graph.add_node(WorkflowNode(name="step1", run=increment_node))
        graph.add_node(WorkflowNode(name="step2", run=always_fails_node))
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        store1 = CheckpointStore(db_path)
        executor1 = WorkflowExecutor(graph, checkpoint_store=store1, max_retries=0)
        first_result = await executor1.run({"count": 0}, make_budget(), run_id="run-1")
        assert first_result.stopped_reason == "failed"
        assert first_result.final_node == "step2"  # the node that failed
        store1.close()

        # A genuinely new executor and checkpoint store, pointed at the same file -
        # this proves resumption reads real persisted state, not in-memory state.
        working_graph = WorkflowGraph(start_node="step1")
        working_graph.add_node(WorkflowNode(name="step1", run=increment_node))
        working_graph.add_node(WorkflowNode(name="step2", run=increment_node))
        working_graph.add_edge("step1", "step2")
        working_graph.add_edge("step2", END)

        store2 = CheckpointStore(db_path)
        executor2 = WorkflowExecutor(working_graph, checkpoint_store=store2)
        resumed_result = await executor2.run({}, make_budget(), run_id="run-1", resume=True)

        assert resumed_result.stopped_reason == "end"
        assert resumed_result.final_state["count"] == 2  # 1 from before restart + 1 after
        store2.close()
