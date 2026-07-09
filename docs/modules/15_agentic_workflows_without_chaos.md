# Module 15 — Agentic Workflows Without Chaos

> Phase: Agents/tools · Bible reference: [curriculum.md §25](../../curriculum.md#25-module-15--agentic-workflows-without-chaos)

## Goal

Teach agentic systems as controlled, inspectable workflows — not open-ended autonomy.

```text
Avoid:
    while True:
        ask LLM what to do next
        run whatever it says

Prefer:
    User request
      -> classify intent
      -> choose workflow
      -> LLM fills specific decision points
      -> deterministic tools execute
      -> validate output
      -> checkpoint state
      -> ask human when needed
```

This module builds **both** shapes for real, on purpose: `react_loop.py` is the "avoid" shape,
deliberately implemented so Lab 2 can break it with adversarial prompts — a real, reproducible
failure, not a hypothetical warning. `workflow_graph.py` is the "prefer" shape, and Lab 3
replaces the broken ReAct loop with it on the exact same task.

> **Machine note:** every planning/execution mechanism here is deterministic Python. The two
> LLM-dependent pieces — the ReAct loop's step-by-step reasoning and a workflow graph node that
> asks the model to fill one specific decision point — are `FakeRuntime`-backed, real adapter
> unchanged later. Everything downstream of an LLM's proposal (loop prevention, safety budgets,
> checkpointing, approval interrupts) is real and runs without a model at all.

## Repo structure note

`packages/local_ai_agents/planners/` (already scaffolded in Phase 0) holds everything that
decides *what to do next* — the ReAct loop, the workflow graph engine, safety budgets, memory,
loop prevention, and checkpointing. `packages/local_ai_agents/executors/` (Module 14's home for
`ToolExecutor`) gains `workflow_executor.py`, which *runs* a workflow graph using Module 14's
`ToolExecutor` for every tool call - planning and execution stay separated in code, not just in
theory (theory doc §2, "Planner-executor pattern").

## Core topics

### 1. Agent vs workflow

An **agent** (this module's "avoid" shape) lets the model decide the next action at every step,
unconstrained. A **workflow** (this module's "prefer" shape) has a fixed set of states and
transitions; the model is only consulted at specific, bounded decision points inside an
otherwise deterministic structure. `react_loop.py` and `workflow_graph.py` implement one of
each, over the identical task, so the difference is measurable (§"Real proof" in the deliverable
report) rather than asserted.

### 2. Planner-executor pattern

`planners/` decides; `executors/` runs. `WorkflowExecutor` never calls a tool handler directly -
every tool call, from either the ReAct loop or a workflow graph node, goes through Module 14's
`ToolExecutor`, so permissions/validation/approval/budgets/audit-logging all still apply
unchanged to agentic tool use.

### 3. ReAct-style loop

`planners/react_loop.py`'s `ReActLoop` - alternates: ask the LLM to reason and propose either a
tool call or a final answer; execute the tool via `ToolExecutor`; append the observation; repeat.
Stops on a final answer, a safety-budget limit, or loop-prevention tripping - Lab 1.

### 4-5. State machine and graph-based agents

One engine, not two: `planners/workflow_graph.py`'s `WorkflowGraph` is a directed graph of named
nodes (each either a deterministic function or a single bounded LLM decision point) connected by
conditional edges. A **state machine is a graph with no branching ambiguity** - the same engine
implements both topics, since a linear chain of states is simply a graph whose conditional edges
each have exactly one true branch. Lab 3's replacement workflow is a `WorkflowGraph` with linear
transitions; a more elaborate graph (multiple conditional edges from one node) would use the
identical engine unchanged.

### 6. Human-in-the-loop

`WorkflowExecutor` checks each node's `requires_approval` flag against Module 14's
`ApprovalGate` before running it - the exact same fail-closed `NullApprovalGate` default, so an
approval-gated workflow node with no real gate wired up halts rather than running. Lab 4.

### 7. Memory

`planners/memory.py`'s `AgentMemory` - the running list of steps (reasoning, tool calls,
observations) within a single workflow run, available to later decision points' prompts.
Explicitly scoped to *within one run*; cross-run/long-term memory is out of scope (that's
RAG-backed memory, Module 11's territory, or Module 8.5's conversation memory - not reinvented
here).

### 8. Tool budgets

Reused unchanged from Module 14: every tool call inside a workflow or ReAct loop still goes
through the same `ToolBudget` via `ToolExecutor`.

### 9. Loop prevention

`planners/loop_prevention.py`'s `LoopGuard` - a real circuit breaker: if the same tool name and
arguments are proposed `max_repeats` times in a row, `LoopGuard` trips and the loop stops with a
`LoopDetectedError`, regardless of what the model wants to do next. This is what actually breaks
Lab 2's adversarial prompt, not a documented risk.

### 10. Failure recovery

`WorkflowExecutor` retries a failed node's action up to `max_retries` times (bounded, not
infinite) before falling back to a configured failure path - a real, tested retry-then-fallback
mechanism, same discipline as Module 6's `with_retries`.

### 11. Deterministic checkpoints

`planners/checkpoint_store.py`'s `CheckpointStore` - real SQLite persistence (same pattern as
Module 8.5's `SessionStore`, Module 14's `AuditLog`): a workflow's current node and memory are
saved after every step, and a new `WorkflowExecutor` can resume a run from its last checkpoint
after an actual process restart, proven not asserted. Lab 5.

### 12. Agent evaluation

`scripts/module_15/evaluate_task_success.py` - a small golden set of (request, expected final
node, expected outcome) cases run through the real `WorkflowGraph`, scored by exact match on the
final state reached - reusing Module 13's evaluation discipline (a golden set + a deterministic
scorer, not a vibe check) rather than building new evaluation infrastructure. Lab 6.

## Agent safety budget

```yaml
max_steps: 8
max_tool_calls: 5
max_runtime_seconds: 60
max_tokens_total: 8000
requires_human_approval:
  - file_write
  - shell_exec
  - db_write
  - network_post
```

Implemented exactly as `planners/safety_budget.py`'s `AgentSafetyBudget` - `max_steps` and
`max_tool_calls` enforced by real counters, `max_runtime_seconds` by real wall-clock elapsed
time (`time.monotonic()`), `max_tokens_total` by summing real `LLMResponse` token counts, and
`requires_human_approval` cross-referenced against each workflow node's tool name.

## Hands-on labs

1. **Implement simple ReAct loop** — `planners/react_loop.py`, `scripts/module_15/react_loop_demo.py`.
2. **Break it with adversarial prompts** — same script: a scripted `FakeRuntime` that always
   proposes the same tool call, provoking `LoopGuard` to trip - a real, reproducible break.
3. **Replace with state-machine workflow** — `planners/workflow_graph.py`,
   `scripts/module_15/workflow_graph_demo.py`, same underlying task as Lab 1, immune to Lab 2's
   adversarial prompt by construction (the model can't choose the next node, the graph's edges
   do).
4. **Add approval interrupt** — `WorkflowExecutor` + Module 14's `ApprovalGate`, same script.
5. **Add checkpointing** — `planners/checkpoint_store.py`, `scripts/module_15/checkpoint_demo.py`.
6. **Evaluate task success** — `scripts/module_15/evaluate_task_success.py`.

## Deliverable

```text
packages/local_ai_agents/
  planners/
    safety_budget.py
    memory.py
    loop_prevention.py
    react_loop.py
    workflow_graph.py
    checkpoint_store.py
    tests/
  executors/
    workflow_executor.py
    tests/ (extended)
scripts/module_15/
  react_loop_demo.py
  workflow_graph_demo.py
  checkpoint_demo.py
  evaluate_task_success.py
reports/module_15_agentic_workflows_report.md
```
