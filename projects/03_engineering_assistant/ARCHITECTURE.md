# Architecture — Project 3: Local Engineering Assistant

> See [PROPOSAL.md](PROPOSAL.md) for why this exists and how success is measured.

## High-level

```text
User request (CLI command + free-text instruction)
  -> eng_intent_classifier.classify_intent()                     [new]
  -> eng_context_builder.build_context()                          [new, composes reused tools]
       -> list_symbols()                                          [reused, Module 17]
       -> search_repo()                                           [reused, Module 17]
       -> read_file_lines()                                       [reused, Module 17]
  -> code model (LLMRuntime.generate, FakeRuntime on this machine) [reused, Module 6]
  -> patch validator
       -> validate_patch_format() + apply_patch()'s context check  [reused, Module 17]
       -> validate_patch_scope()                                   [new]
       -> validate_hunk_line_counts()                               [new]
  -> human approval (ApprovalGate, via ToolExecutor)                [reused, Module 14]
  -> patch applier (apply_patch, via eng_tools.py's ToolExecutor)   [reused + newly wired]
  -> test runner (run_tests, via eng_tools.py's ToolExecutor)       [reused + newly wired]
```

**Deployment shape**: a real `typer` CLI (`eng_cli.py`), curriculum's own "CLI: best for
developer tools and labs" framing — no FastAPI service for this project (a deliberate choice,
documented, not a gap: curriculum gives Projects 1/2 explicit API sketches but gives Project 3
none, and its own deployment-modes table names CLI as the right fit for this exact use case). No
new persistent storage beyond the `AuditLog` Module 14 already provides — every tool call this
project makes is logged there.

**Reused components, exact source**:

| Component | Source | Role here |
|---|---|---|
| `list_symbols`, `search_repo`, `read_file_lines` | `local_ai_agents/tools/{list_symbols,search_repo,read_file}.py` | repo indexing, code search, file reads |
| `propose_patch`, `validate_patch_format`, `apply_patch` | `local_ai_agents/tools/patch_tools.py` | patch proposal, format validation, sandboxed application |
| `run_tests` | `local_ai_agents/tools/run_tests.py` | sandboxed pytest execution |
| `ToolExecutor` | `local_ai_agents/executors/tool_executor.py` | permission → approval → budget → audit-log pipeline for every tool call |
| `ApprovalGate`, `PermissionPolicy`, `ToolBudget`, `AuditLog` | `local_ai_agents/policies/*.py` | Module 14's enforcement stack |
| `resolve_within_sandbox`, `PathTraversalError` | `local_ai_agents/tools/sandbox.py` | the real defense against "model invents a file path" |
| `with_timeout` | `local_ai_core/security/tool_call_timeout.py` | uniform per-call timeout, applied to every tool here (not just `run_tests`'s own bespoke one) |
| `AppContext`, `build_app_context` | `local_ai_core/deployment/app_context.py` | composition root |

**New components** (why nothing existing covers them — see PROPOSAL.md's survey): intent
classification, context assembly, patch-scope/line-count validation, command-safety allowlisting,
`Tool` registrations for the patch functions, code-specific prompts, and a richer demo repo.

## Low-level

### Intent classification (`eng_intent_classifier.py`)

```python
IntentType = Literal[
    "explain_repo", "search_code", "explain_symbol",
    "generate_tests", "suggest_refactor", "propose_patch", "run_tests",
]

def classify_intent(request: str) -> IntentClassification:  # (intent, reason)
```

Real keyword/pattern matching over the request text — the same discipline Module 18's
`should_use_vlm()` and Module 20's `route_model()` applied to their own routing decisions: a
traceable decision with a stated reason, not a vibe, and no model call needed for something this
deterministic.

### Context building (`eng_context_builder.py`)

```python
@dataclass(frozen=True)
class ContextBundle:
    intent: IntentType
    repo_symbols: dict[str, list[Symbol]]     # per-file, from list_symbols()
    search_results: list[RepoMatch]           # from search_repo(), only for search_code intent
    file_excerpt: str | None                  # from read_file_lines(), for explain_symbol/propose_patch

def build_context(allowed_base: Path, intent: IntentType, *, query: str | None = None, target_file: str | None = None, symbol_name: str | None = None) -> ContextBundle
```

For `explain_repo`: symbols across every `.py` file under `allowed_base` (via `search_repo`-style
`rglob("*.py")` + `list_symbols` per file). For `explain_symbol`/`propose_patch`: symbols for
`target_file` plus the specific line range `read_file_lines` returns once `symbol_name`'s line is
located. For `search_code`: `search_repo(query)`'s raw matches, no symbol listing needed.

### Patch scope and line-count validation (`eng_patch_guard.py`)

```python
class PatchScopeError(Exception): ...       # new, this project's own - not Module 17's PatchFormatError
class PatchLineCountError(Exception): ...    # new

def validate_patch_scope(parsed: ParsedPatch, expected_file_path: str) -> None
def validate_hunk_line_counts(patch_text: str) -> None   # re-parses @@ headers' `,N` counts, ignored by patch_tools.py's regex
```

Both run **before** `apply_patch()` is ever called — curriculum's own "validate patch before
applying" functional requirement, made real for the two specific gaps `patch_tools.py`'s own
validation doesn't cover (confirmed by reading its regex: the hunk-header pattern's count groups
are non-capturing, and there is no check anywhere that a patch's single `file_path` matches what
the request was actually about).

### Command safety (`eng_command_safety.py`)

```python
ALLOWED_TEST_COMMAND_PREFIX = (sys.executable, "-m", "pytest")

class UnsafeCommandError(Exception): ...

def validate_test_command(argv: list[str]) -> None
```

`run_tests()`'s real invocation is already fixed and not model-controlled (confirmed by reading
`run_tests.py`) — this validator exists for the one place curriculum's "unsafe shell command"
failure case could actually reach code: a hypothetical model-suggested test-run command, checked
against an exact-prefix allowlist before it would ever be allowed to run, never a free-form shell
string.

### Tool registrations (`eng_tools.py`)

```python
def make_propose_patch_tool(runtime: LLMRuntime, model: str) -> Tool
def make_apply_patch_tool(allowed_base: Path, *, expected_file_path_provider: Callable[[dict], str]) -> Tool
```

Wraps `patch_tools.propose_patch`/`apply_patch` (previously bare functions with no `Tool`
wrapper, confirmed by survey) as real `Tool` objects with Pydantic arg models, registered into a
`ToolRegistry` and called only through `ToolExecutor` — the same audit-logging/permission/
approval/budget path every other tool in this project uses, closing the gap where patch
operations previously bypassed it. `apply_patch`'s tool wraps the call in `with_timeout()`
(Module 22) in addition to `ToolExecutor`'s own approval gate, since applying a patch touches the
filesystem and deserves the same time-bound as running tests does.

### Composition root (`eng_service.py`)

```python
@dataclass
class EngAppContext:
    base: AppContext
    repo_dir: Path
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor

def build_eng_context(config: AppConfig, *, model_catalog_path, repo_dir: Path, runtime: LLMRuntime | None = None, approval_gate: ApprovalGate | None = None) -> EngAppContext
```

Same extension pattern Projects 1 and 2 established: wraps `AppContext`, adds project-specific
state (`repo_dir`, a `ToolRegistry` with every capability registered, a `ToolExecutor` wired with
`base.audit_log`). `approval_gate` defaults to `NullApprovalGate` (fail-closed — Module 14's own
safe default), so a caller must explicitly opt into `AutoApprovalGate`/`CallbackApprovalGate` to
let `apply_patch`/`run_tests` actually run.

### CLI (`eng_cli.py`)

A real `typer.Typer()` app (same pattern Module 23's `cli_app.py` established): `explain-repo`,
`search`, `explain-symbol`, `generate-tests`, `suggest-refactor`, `propose-patch`,
`apply-patch`, `run-tests` commands, each routing through `EngAppContext`'s `ToolExecutor`.

### The demo repo (`demo_repo/`)

A small, real, multi-file inventory-management package (distinct from Module 17's
`mini_calculator`, deliberately richer per that module's own report — "a second scenario would
exercise the same code path, not new logic"):

```text
demo_repo/
  README.md
  inventory/
    __init__.py
    stock.py       # StockItem, add_stock(), remove_stock() - real bug: no validation against
                    # removing more than current quantity
    pricing.py      # calculate_discount(), apply_tax() - no bug, cross-file dependency target
    reports.py       # generate_summary() - imports from both stock.py and pricing.py
  tests/
    test_stock.py    # includes one currently-FAILING test proving the real bug
    test_pricing.py
    test_reports.py
```

Multiple files (exercises multi-file symbol search and gives `validate_patch_scope()` a real
"wrong file" case to reject against), a cross-module dependency (`reports.py` importing from two
siblings, for the "misses a dependency/import" failure case), and one real, currently-failing
test — proven by running the demo repo's own suite unmodified (see REPORT.md).
