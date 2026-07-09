# Module 14 — Tool Calling and Deterministic Tool Execution

> Phase: Agents/tools · Bible reference: [curriculum.md §24](../../curriculum.md#24-module-14--tool-calling-and-deterministic-tool-execution)

## Goal

Build safe tool-calling systems where the LLM *proposes* and deterministic code *enforces*.

```text
The LLM may decide:
    I want to call search_files with argument query="auth middleware"

But deterministic code decides:
    Is this tool allowed?
    Are arguments valid?
    Is user authorized?
    Is approval required?
    Can the tool access this path?
    How much data can be returned?
```

Every one of those six questions gets real, tested code this module — none of them are left to
"the model will probably behave."

> **Machine note:** unlike most RAG modules, almost everything here is deterministic Python with
> no model dependency at all — schema validation, the registry, permissions, approval gating,
> audit logging (real SQLite, same pattern as Module 8.5's `SessionStore`), and budgets all run
> for real. The one LLM-dependent piece is `tool_call.py`'s `propose_tool_call()` (asking a model
> which tool to call) — `FakeRuntime`-backed here, real adapter unchanged later.

## Repo structure note

`packages/local_ai_agents/tools/`, `policies/`, and `executors/` follow curriculum.md §8's
canonical structure (already scaffolded in Phase 0). `policies/` owns the four deterministic
gates (permissions, approval, budgets, audit log) as separate, independently testable modules
rather than one large "safety" file — each gate answers exactly one of the tool execution rule's
six questions.

## Core topics

### 1. Tool calling mental model

The LLM's output is a *proposal*, never a command. `ToolCallProposal` (`tools/tool_call.py`) is
inert data — a tool name and arguments — until `ToolExecutor` (`executors/tool_executor.py`)
decides whether to actually run it. No code path in this repo calls a tool handler directly from
an LLM response.

### 2. Function schemas

`tools/base.py`'s `Tool` dataclass pairs a Pydantic `BaseModel` (the args schema) with a handler
function - curriculum's own `SearchFilesArgs` example, implemented for real in `file_search.py`.
`Tool.json_schema()` renders the Pydantic model's schema for exposing to an LLM prompt (`tool_call.py`).

### 3. Tool registry

`tools/registry.py`'s `ToolRegistry` - register, look up by name, and list every registered
tool's schema. A tool call for an unregistered name fails at the registry, before validation or
execution ever run.

### 4. Tool selection

`tool_call.py`'s `propose_tool_call()` - one LLM call, given the registry's tool schemas and a
user request, asked to respond with a JSON tool-call proposal. Parsed strictly (real
`ToolCallParseError` on malformed JSON, not a silent fallback) - the same honesty standard
Module 8's structured-output reliability ladder established.

### 5. Argument validation

Every `Tool.handler` call goes through `args_model.model_validate(arguments)` first
(`ToolExecutor`) - a genuinely invalid argument (wrong type, out-of-range value, curriculum's own
`max_results: int = Field(ge=1, le=50)` constraint) raises `ToolValidationError` before the
handler ever runs, not a runtime crash inside the tool.

### 6. Tool result formatting

`base.py`'s `ToolResult` - a uniform `success`/`data`/`error_message` shape every tool returns,
plus `ToolResult.as_text()` for feeding back into a conversation - so a caller never has to
special-case each tool's own return shape.

### 7. Tool error handling

A real error taxonomy, mirroring Module 6's `LLMError` hierarchy: `ToolNotFoundError`,
`ToolValidationError`, `ToolPermissionError`, `ToolApprovalRequiredError`, `ToolBudgetExceededError`,
`ToolExecutionError` (the handler itself raised) - `ToolExecutor` catches each distinctly and
returns a `ToolResult(success=False, ...)` rather than letting any of them escape uncaught.

### 8. Permissions

`policies/permissions.py`'s `PermissionPolicy` - real role-based allow/deny (a role either can
or cannot call a given tool name), checked before a tool ever runs, not inferred from what the
model says it's allowed to do.

### 9. Human approval

`policies/approval.py`'s `ApprovalGate` - dangerous tools require an injected async approval
callback to return `True` before execution. `NullApprovalGate` always denies (a dangerous tool
call from a caller that never wired up real approval fails closed, not open) and
`AutoApprovalGate` (tests only, explicitly documented as unsafe for real use) always approves.

### 10. Dangerous tools

`Tool.dangerous: bool` flag, set `True` on curriculum's own named categories present in this
module's tools (writing files). `ToolExecutor` checks `dangerous` before `ApprovalGate`, so a
non-dangerous tool never needs an approval callback wired up at all.

### 11. Audit logging

`policies/audit_log.py`'s `AuditLog` - real SQLite persistence (stdlib `sqlite3`, no server),
same pattern as Module 8.5's `SessionStore`: every tool call attempt (allowed or denied,
succeeded or failed) is logged with a trace id, arguments, outcome, and timestamp, proven across
an actual close/reopen cycle, not asserted from an in-memory list.

### 12. Tool budgets

`policies/budgets.py`'s `ToolBudget` - per-session and per-tool call limits, real counters that
raise `ToolBudgetExceededError` once exhausted, same "never trust the model to stop calling
tools on its own" discipline the ReAct-loop-prevention topic in Module 15 will build on.

## Tool schema example

```python
from pydantic import BaseModel, Field

class SearchFilesArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    root_path: str = Field(default=".")
    max_results: int = Field(default=10, ge=1, le=50)
```

Implemented verbatim in `file_search.py`.

## Dangerous tools

Curriculum's own list (writing files, deleting files, executing shell commands, sending emails,
making network calls, modifying databases, committing code, deploying services, accessing
secrets). This module implements one representative dangerous tool (`write_file.py`, gated by
real human approval) rather than all nine categories - the *mechanism* (dangerous flag ->
approval gate -> audit log) is what's being proven, and it's identical regardless of which
dangerous category triggers it.

## The four tools built this module

| Tool | Dangerous | Real safety mechanism |
|---|---|---|
| `calculator.py` | No | AST-based expression evaluation - only numeric literals and `+ - * / ** ()` nodes are allowed; any other AST node (`Name`, `Call`, `Attribute`, …) is rejected before evaluation, so this is not Python `eval()` in a trenchcoat |
| `file_search.py` | No | Path containment - `root_path` is resolved and every candidate file's resolved path must be a descendant of it, rejecting `..`-style traversal outside the sandboxed root |
| `sql_query.py` | No | Read-only SQLite connection (`mode=ro` URI) *and* a query-level check rejecting anything but a single `SELECT` statement - defense in depth, not just one layer |
| `write_file.py` | **Yes** | Path containment (same as `file_search.py`) *and* mandatory `ApprovalGate` approval before any write happens |

## Hands-on labs

1. **Build tool registry** — `tools/registry.py`, `scripts/module_14/tool_registry_demo.py`.
2. **Add a safe calculator tool** — `tools/calculator.py`.
3. **Add file search tool** — `tools/file_search.py`, searches `datasets/rag_docs/nimbus_handbook/`.
4. **Add a SQL read-only tool** — `tools/sql_query.py`, real SQLite fixture database.
5. **Add human approval for write tool** — `tools/write_file.py` + `policies/approval.py`,
   `scripts/module_14/approval_and_dangerous_tools_demo.py`.
6. **Add tool audit logs** — `policies/audit_log.py`, same script.

## Deliverable

```text
packages/local_ai_agents/
  tools/
    base.py
    registry.py
    tool_call.py
    calculator.py
    file_search.py
    sql_query.py
    write_file.py
    tests/
  policies/
    permissions.py
    approval.py
    budgets.py
    audit_log.py
    tests/
  executors/
    tool_executor.py
    tests/
scripts/module_14/
  tool_registry_demo.py
  approval_and_dangerous_tools_demo.py
reports/module_14_tool_calling_report.md
```
