"""WorkflowExecutor - runs a `WorkflowGraph` deterministically: safety
budget enforcement per step, human-approval gating for nodes that need it
(Module 14's `ApprovalGate`, same fail-closed default), bounded retry then
fallback on node failure, and a checkpoint saved after every successful
step so a run can resume after an actual process restart.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from local_ai_agents.planners.checkpoint_store import CheckpointStore
from local_ai_agents.planners.memory import AgentMemory
from local_ai_agents.planners.safety_budget import AgentSafetyBudget, SafetyBudgetExceededError
from local_ai_agents.planners.workflow_graph import END, WorkflowGraph, WorkflowNode, WorkflowState
from local_ai_agents.policies.approval import ApprovalGate, NullApprovalGate


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    #: The node the run *stopped at* - not necessarily one that finished
    #: successfully. On "end" this is the graph's `END` sentinel; on
    #: "safety_budget" it's the node the budget check fired in front of
    #: (possibly never run); on "approval_denied"/"failed" it's the node
    #: that was denied/failed. Consistent across every stop reason: this
    #: always answers "where did the run stop", not "what last succeeded".
    final_node: str
    final_state: WorkflowState
    stopped_reason: str  # "end" | "safety_budget" | "approval_denied" | "failed"
    memory: AgentMemory
    steps_taken: int


class WorkflowExecutor:
    def __init__(
        self,
        graph: WorkflowGraph,
        *,
        approval_gate: ApprovalGate | None = None,
        checkpoint_store: CheckpointStore | None = None,
        max_retries: int = 2,
    ) -> None:
        self._graph = graph
        self._approval_gate: ApprovalGate = approval_gate or NullApprovalGate()
        self._checkpoint_store = checkpoint_store
        self._max_retries = max_retries

    async def run(
        self,
        initial_state: WorkflowState,
        safety_budget: AgentSafetyBudget,
        *,
        run_id: str | None = None,
        resume: bool = False,
    ) -> WorkflowResult:
        run_id = run_id or str(uuid.uuid4())
        memory = AgentMemory()
        current_node, state, step_index = self._graph.start_node, dict(initial_state), 0

        if resume and self._checkpoint_store is not None:
            checkpoint = self._checkpoint_store.load(run_id)
            if checkpoint is not None:
                current_node, state, step_index = checkpoint.node_name, dict(checkpoint.state), checkpoint.step_index
                memory.add("node_transition", f"resumed run '{run_id}' from checkpointed node '{current_node}'")

        while current_node != END:
            try:
                safety_budget.record_step()
                safety_budget.check_runtime()
            except SafetyBudgetExceededError:
                return WorkflowResult(run_id, current_node, state, "safety_budget", memory, step_index)

            node = self._graph.get_node(current_node)

            if node.requires_approval_tool is not None:
                approved = await self._approval_gate.request_approval(node.requires_approval_tool, state)
                memory.add("node_transition", f"approval for '{node.requires_approval_tool}': {approved}")
                if not approved:
                    return WorkflowResult(run_id, current_node, state, "approval_denied", memory, step_index)

            new_state = await self._run_node_with_retries(node, state, memory)
            if new_state is None:
                return WorkflowResult(run_id, current_node, state, "failed", memory, step_index)
            state = new_state

            memory.add("node_transition", f"completed node '{current_node}'")
            step_index += 1

            next_node = self._graph.next_node(current_node, state)
            if next_node is None:
                return WorkflowResult(run_id, current_node, state, "failed", memory, step_index)

            # Checkpoint the *next* node, not the one that just completed -
            # resuming must continue forward, never re-run a node that
            # already finished (a real bug caught by the resume test: an
            # earlier version checkpointed `current_node` here, which made
            # resume re-execute the just-completed node a second time).
            if self._checkpoint_store is not None:
                self._checkpoint_store.save(run_id, next_node, state, step_index)
            current_node = next_node

        return WorkflowResult(run_id, current_node, state, "end", memory, step_index)

    async def _run_node_with_retries(
        self, node: WorkflowNode, state: WorkflowState, memory: AgentMemory
    ) -> WorkflowState | None:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await node.run(state, memory)
            except Exception as exc:  # noqa: BLE001 - real, bounded retry-then-fail, never swallowed
                last_error = exc
                memory.add("node_transition", f"node '{node.name}' attempt {attempt + 1} failed: {exc}")
        memory.add(
            "node_transition", f"node '{node.name}' failed after {self._max_retries + 1} attempts: {last_error}"
        )
        return None
