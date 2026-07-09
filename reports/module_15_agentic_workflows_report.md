# Module 15 deliverable — agentic workflows report

Status: **complete.** Every planning/execution mechanism is deterministic Python and runs for
real — safety budgets, loop prevention, the workflow graph engine, checkpointing, and human
approval interrupts. The two LLM-dependent pieces (ReAct's step-by-step reasoning, one bounded
workflow decision point) are `ScriptedTurnRuntime`/`FakeRuntime`-backed; real model quality is
deferred to the resourced 32GB Mac, same discipline as every module since 9.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `planners/safety_budget.py` | 9 | Real step/tool-call/token counters and wall-clock runtime checks, all raising once exceeded |
| `planners/memory.py` | 6 | Ordered step history, transcript rendering |
| `planners/loop_prevention.py` | 6 | Circuit breaker trips on identical repeated calls, resets on a different call, order-independent argument signatures |
| `planners/react_loop.py` | 5 | Final-answer stop, tool-call-then-answer flow, safety-budget stop, loop-detected stop |
| `planners/workflow_graph.py` | 8 | Linear traversal, conditional branching, misconfiguration errors |
| `planners/checkpoint_store.py` | 6 | Real SQLite persistence, proven across an actual close/reopen cycle |
| `executors/workflow_executor.py` | 24 | Full graph execution: safety budget, approval gating, bounded retry-then-fail, checkpointing and resume — including a real bug caught and fixed (see below) |
| `scripts/module_15/` (4 lab scripts) | 73 total incl. above | Labs 1-6 exercised on a real SQLite fixture database and real sandboxed file writes |
| `notebooks/15_agentic_workflows_without_chaos.ipynb` | — | **Executed end-to-end** — every cell a real measurement |

**73 new tests this module** (1345 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: the "avoid" shape actually breaks (from the executed notebook)

Same task, same corpus, two implementations:

```
Lab 1 (ReAct, happy path):    stopped_reason=final_answer, answer="There are 3 open tickets."
Lab 2 (ReAct, adversarial):   stopped_reason=loop_detected, LLM calls made before trip: 3
```

An adversarial scripted runtime that always re-proposes the identical `sql_query` call never
produces a final answer on its own — `LoopGuard` is what actually stops it, at exactly the
`max_repeats=3` threshold configured, well before the `max_steps=8` safety budget would have
caught it. This is a real, reproducible failure of the "avoid" shape, not a documented risk.

## Real proof: the "prefer" shape is immune by construction, not by catching the same failure later

The identical adversarial runtime, run against `workflow_graph_demo.py`'s deterministic graph
for the identical task:

```
Same adversarial runtime from Lab 2, run against the graph: end (only 1 LLM call(s) made - no loop possible)
```

Not "LoopGuard caught it here too" — there is no step in the graph where the model chooses a
tool or could repeat an action. The model fills exactly one bounded decision point (wording the
summary) and the graph's deterministic edges decide everything else. `query_tickets`'s tool call
is hardcoded, not model-proposed, so an adversarial prompt has nothing to influence.

## Real proof: human approval interrupt, with the write actually verified on disk

```
Dangerous node denied by default (no approval gate): approval_denied
Dangerous node approved: end -> There are 3 open tickets requiring attention.
summary.txt actually written to disk: True
```

Same fail-closed default as Module 14's `ApprovalGate`: a dangerous workflow node
(`log_summary`, `requires_approval_tool="write_file"`) halts the entire run when no real
approval gate is configured, and `summary.txt`'s existence on disk was checked directly — not
just that the `ToolResult` claimed success.

## Real proof: checkpointing survives an actual restart, and doesn't redo finished work

```
First run stopped at 'categorize' (failed) with state {'tickets_fetched': 5}
Resumed run (new executor, new store, same SQLite file) result: end
Final state after resume: {'tickets_fetched': 5, 'categorized': True, 'done': True}
```

A genuinely new `WorkflowExecutor` and `CheckpointStore`, pointed at the same SQLite file after
the first run failed at `categorize` — not a mock, an actual file read from disk. `fetch`'s
result (`tickets_fetched: 5`) survived into the resumed run without `fetch` re-executing.

## A real bug caught and fixed during development

The first version of `WorkflowExecutor` saved a checkpoint using the node that had *just
completed* as the resume point. On resume, that made the executor **re-run the already-completed
node a second time** instead of continuing forward — caught by
`test_resuming_continues_from_the_last_checkpointed_node`, which expected a counter to be `2`
after resume (1 before the simulated restart, 1 after) and got `3` (the completed node ran
twice: once before the restart, once again after resume). Fixed by checkpointing the *next*
node computed from the graph's edges, not the node that just finished — documented directly in
the executor's source as the reason for that ordering, not just in this report.

## Real proof: agent task success evaluation actually discriminates

```
Task success rate: 2/3
- normal_run: success=True
- verbose_summary: success=True
- deliberately_wrong_expectation: success=False
```

A third golden case with a deliberately wrong expected ticket count (99 instead of the real 3)
was added specifically to prove the scorer fails a case it should fail, not just report 100%
because every case happens to succeed — the same non-rubber-stamp discipline established in
Module 11's honest recall numbers.

## Deliberately not done in Module 15

- No real LLM driving the ReAct loop's reasoning or the workflow graph's summary node — both
  are fully built and unit-tested with scripted runtimes; real model behavior (does a small
  local model actually get stuck in the ReAct loop the way the adversarial script does on
  purpose? does it fill the one bounded decision point sensibly?) is deferred to the resourced
  32GB Mac.
- Cross-run/long-term agent memory — `AgentMemory` is explicitly scoped to a single run;
  persistent memory across runs is RAG-backed memory (Module 11) or conversation memory
  (Module 8.5), not reinvented here.
- Only one dangerous workflow node demonstrated (`log_summary`/`write_file`) — the mechanism
  (node-level `requires_approval_tool` → `ApprovalGate`) is identical regardless of which tool
  triggers it, consistent with Module 14's same scoping decision.
