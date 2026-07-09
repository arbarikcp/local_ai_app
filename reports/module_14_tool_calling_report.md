# Module 14 deliverable — tool calling and deterministic execution report

Status: **complete.** Almost no honest-skip surface at all this module — schema validation, the
tool registry, permissions, human approval gating, tool budgets, and real SQLite-backed audit
logging are all deterministic Python with zero model dependency, and run for real. The one
LLM-dependent piece, `propose_tool_call()` (asking a model which tool to call), is
`FakeRuntime`-backed; the deterministic enforcement layer behind it doesn't change at all once a
real model is swapped in.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `tools/base.py` | 9 | `Tool`/`ToolResult`/`ToolCallProposal` shapes, JSON schema rendering, dangerous flag |
| `tools/registry.py` | 7 | Register/get/list/schema_list, `ToolNotFoundError` for unregistered names |
| `tools/tool_call.py` | 5 | Strict JSON parsing of an LLM's proposed tool call, real parse-failure detection |
| `tools/sandbox.py` | 6 | Path containment — including two real attack vectors (`..` traversal, absolute-path override) |
| `tools/calculator.py` | 15 | Correct arithmetic *and* 7 real code-injection payloads rejected before evaluation |
| `tools/file_search.py` | 8 | Filename/content search, sandboxed, traversal rejected |
| `tools/sql_query.py` | 13 | Query-text validation *and* SQLite's own read-only connection independently rejecting writes |
| `tools/write_file.py` | 7 | Sandboxed writes, `dangerous=True`, traversal and absolute-path rejection |
| `policies/permissions.py` | 5 | Role-based allow/deny, wildcard grants |
| `policies/approval.py` | 4 | Fail-closed default, callback delegation, auto-approve (tests only) |
| `policies/budgets.py` | 7 | Total and per-tool call limits, independent per-tool counters |
| `policies/audit_log.py` | 8 | Real SQLite persistence, proven across an actual close/reopen cycle |
| `executors/tool_executor.py` | 14 | Full enforcement chain: registry → permissions → validation → approval → budget → handler → audit log, every stage independently testable |
| `scripts/module_14/` (2 lab scripts) | 17 | Labs 1-6 exercised against the real Nimbus handbook corpus and a real SQLite fixture database |
| `notebooks/14_tool_calling_and_deterministic_execution.ipynb` | — | **Executed end-to-end** — every cell a real measurement |

**124 new tests this module** (1272 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: the calculator is not `eval()` in a trenchcoat (from the executed notebook)

```
(2 + 3) * 4 = 20
2 ** 10 = 1024
Rejected: "__import__('os').system('echo pwned')" -> Disallowed expression element: Call
Rejected: "os.system('ls')" -> Disallowed expression element: Call
Rejected: '(1).__class__' -> Disallowed expression element: Attribute
```

`safe_eval()` walks the parsed AST and only permits numeric literals and arithmetic operators —
every one of three real code-injection payloads is rejected by AST node type, not by a regex
blocklist that a sufficiently creative payload could evade.

## Real proof: path containment catches a pathlib gotcha most implementations miss

```
Rejected: '../../../etc/passwd' -> resolves outside the allowed sandbox .../nimbus_handbook
Rejected: '/etc/passwd' -> resolves outside the allowed sandbox .../nimbus_handbook
```

The second case is the more interesting one: `Path(base) / "/etc/passwd"` under plain pathlib
semantics **silently discards `base` entirely** and returns `/etc/passwd` — a well-known,
easy-to-miss footgun. `resolve_within_sandbox()` catches it anyway, because containment is
checked *after* the join, not assumed from how the join was constructed. Verified with an
explicit test (`test_absolute_path_override_is_rejected`), not just the traversal case.

## Real proof: the SQL tool has two independent defense layers, not one

```
Layer 1 (query-text check) rejected DELETE: Only SELECT statements are allowed.
Layer 2 (SQLite's own read-only mode) independently rejected it too: attempt to write a readonly database
```

Layer 2 was verified by bypassing this tool's own validation entirely and opening the exact same
`mode=ro` connection string directly — proving SQLite's own read-only enforcement works
independently of this tool's Python-level checks, not merely asserting "defense in depth" as a
design intent.

## Real proof: the full enforcement chain, end to end (from the executed notebook)

```
Denied by default (no real approval gate configured): True
Approved write ('notes.txt', callback approves): True
Denied write ('secrets.txt', callback denies): True
Denied by permissions before approval was even asked ('guest' role): True
Budget test outcomes (3 attempts, 2-call budget): ['success', 'success', 'denied']
Files actually written to the sandbox: ['notes.txt', 'notes_0.txt', 'notes_1.txt']
Total audit log entries recorded: 7
```

Every stage of curriculum's own tool execution rule is independently demonstrated: a dangerous
tool with no approval gate wired up fails closed (`NullApprovalGate`); a real (if scripted)
approval callback approves some calls and denies others based on the actual argument content,
not a hardcoded True/False; a role never granted `write_file` is denied *before* approval is
even asked (permissions checked first, cheaper failure earlier); a budget genuinely stops
execution after its limit, mid-sequence; and `secrets.txt` — the denied write — never appears on
disk, confirming the deny actually prevented the filesystem write rather than just returning a
denied `ToolResult` while the handler ran anyway.

## Real proof: audit log persistence survives an actual restart (from the executed notebook)

```
Closed log1 (simulating a process restart)...
Entries recovered after restart: 1
  2026-07-09T11:32:41.699072+00:00  calculator  success  returned 4
```

A genuinely new `AuditLog` instance, opened against the same SQLite file after the first
instance was closed — not a mock, not the same Python object, an actual file read from disk.
Same proof standard as Module 8.5's `SessionStore` restart test.

## A real bug caught during development (not shipped)

Early in `packages/local_ai_agents/tools/tests/test_base.py`, pytest's test collection failed
with a module-name collision against `packages/local_ai_core/runtimes/tests/test_base.py` (both
files were named `test_base.py`, and neither package uses `__init__.py` markers in its `tests/`
directory, so pytest's default rootdir-relative import can't disambiguate them). Renamed to
`test_tool_base.py` and `test_tool_registry.py` (the latter colliding with
`local_ai_core/prompts/tests/test_registry.py`) before running the full suite — the same fix
Module 11's `test_pipeline.py` needed for the same underlying reason.

## Deliberately not done in Module 14

- No real LLM proposing tool calls — `propose_tool_call()` is fully built and unit-tested with
  `FakeRuntime`; real model quality (does a 1.5B-4B model reliably produce valid tool-call JSON?)
  is deferred to the resourced 32GB Mac.
- Only one dangerous tool implemented (`write_file`) rather than all nine of curriculum's named
  dangerous categories (deleting files, shell execution, email, network calls, database writes,
  code commits, deployments, secrets access) — the *mechanism* (dangerous flag → approval gate →
  audit log) is identical regardless of which category triggers it, so a second or third
  dangerous tool would exercise the same code path, not new logic.
- No tool-result size limiting ("how much data can be returned?" from the tool execution rule) —
  `SearchFilesArgs.max_results` and `SqlQueryArgs.max_rows` already bound the two tools whose
  output size could otherwise be unbounded; a generic result-truncation layer in `ToolExecutor`
  itself wasn't built since both existing tools already self-limit at the schema level.
- `ToolExecutor` doesn't distinguish `ToolValidationError`/`ToolExecutionError` when a handler
  itself raises a domain-specific error (e.g. `sql_query.py`'s `UnsafeQueryError` surfaces as a
  generic wrapped execution failure, not a distinct error type) — documented as a real,
  acceptable simplification: the audit log and `ToolResult.error_message` still capture the
  real cause, just not as a separately-typed exception a caller could branch on.
