# Proposal — Project 3: Local Engineering Assistant

> Bible reference: [curriculum.md §36](../../curriculum.md#36-project-3--local-engineering-assistant) · Structure convention: [projects/PROJECT_TEMPLATE.md](../PROJECT_TEMPLATE.md)

## Why

A repo-aware coding assistant is the third central production use case this course builds
toward, after structured extraction (Project 1) and RAG (Project 2). Module 17 already built and
proved every individual tool a coding assistant needs — AST-based symbol listing, repo search,
sandboxed file reading, unified-diff proposal/validation/application, and sandboxed test
running — wired into one real, tested `WorkflowGraph` pipeline (`coding_assistant.py`). Project 3
is where that pipeline becomes a product a developer could actually run against their own repo
from a terminal: an intent-routed CLI, a richer demo repository that exercises failure modes
Module 17's 2-file sample never needed to, and the two real safety gaps Module 17's own report
left open (unrelated-file-change detection, unsafe-command rejection).

## How

**Reused, not rebuilt** (confirmed by survey):

- `local_ai_agents/tools/{list_symbols,search_repo,read_file}.py` — real AST-based symbol
  listing, lexical repo search, sandboxed line-range reads. Unchanged.
- `local_ai_agents/tools/patch_tools.py` — `propose_patch()` (the one real LLM call),
  `validate_patch_format()`/`apply_patch()` (real unified-diff parsing, context-mismatch
  rejection). Unchanged; Project 3 wraps additional checks around it rather than modifying it.
- `local_ai_agents/tools/run_tests.py` — real sandboxed `pytest` subprocess execution with its
  own timeout. Unchanged.
- `local_ai_agents/executors/tool_executor.py`'s `ToolExecutor`, `policies/{approval,audit_log,
  permissions,budgets}.py` — the full Module 14 enforcement stack (permission check, approval
  gate, budget, audit log) every tool call in this project routes through.
- `local_ai_core/security/tool_call_timeout.py`'s `with_timeout()` — Module 22's generic
  per-call timeout wrapper, applied uniformly here (confirmed by survey: previously only
  `run_tests`'s own bespoke `subprocess` timeout existed; nothing else in the coding-assistant
  path was time-bounded).
- `local_ai_core/deployment/app_context.py`'s `AppContext`/`build_app_context()` — Module 23's
  composition root, extended the same way Projects 1 and 2 extended it.
- `local_ai_agents/tools/sandbox.py`'s `resolve_within_sandbox()` — every file operation in this
  project stays inside `demo_repo/` (or whatever repo path a caller configures), never touches
  the rest of the filesystem.

**Built fresh** (confirmed, by survey, that nothing in the repo already does this):

- `app/eng_intent_classifier.py` — nothing in the repo classifies a free-text request into a
  task type; Module 17's own pipeline is single-shaped, driven by a caller-supplied instruction.
- `app/eng_patch_guard.py` — two real gaps in `patch_tools.py`'s validation, confirmed by
  reading the code: no check that a patch's target file matches the file the request was
  actually about (curriculum's "model changes unrelated files" failure case), and no check that
  a hunk's `@@` line-count header matches its actual body (a malformed-but-parseable patch could
  slip through).
- `app/eng_command_safety.py` — curriculum names "model suggests unsafe shell command" as a
  failure case to test; `run_tests.py`'s pytest invocation is already fixed and not
  model-controlled, so this project adds a real allowlist check for the one place a model's
  suggestion *could* reach a command (a hypothetical model-proposed test-run command), rather
  than leaving the failure case untestable.
- `app/eng_tools.py` — `patch_tools.py`'s functions have no `Tool`/registry wrappers today
  (`make_apply_patch_tool`/`make_propose_patch_tool` don't exist), so calling them bypasses
  `ToolExecutor`'s audit-logging/permission/budget layer entirely. This project adds them.
- `app/eng_context_builder.py` — no code assembles symbol lists + search results + file excerpts
  into one context bundle for a given task; Module 17's `coding_assistant.py` does this ad hoc,
  inline, for one fixed pipeline shape only.
- `prompts/eng_prompts.py` — "explain a function/class" and "suggest refactoring" have no prompt
  templates anywhere (only `patch_tools.py`'s `PATCH_PROMPT_TEMPLATE` and a script-inline
  test-generation template exist).
- `demo_repo/` — Module 17's `datasets/code_repos/mini_calculator/` is a 2-file fixture,
  explicitly named in its own report as too small to exercise multi-file patches, "unrelated
  file changes," or a richer repo map. This project needs a fixture repo built for that purpose.

## What this achieves

A real `typer` CLI (`app/eng_cli.py`, curriculum's own "CLI: best for developer tools and labs"
framing) that:

1. Explains repo structure (`explain-repo` — real symbol map across every file).
2. Searches code (`search` — real lexical search, real line numbers).
3. Explains a function/class (`explain-symbol` — real AST-located source + a real LLM call).
4. Generates tests (`generate-tests` — a new, real prompt template + the reused LLM-call pattern).
5. Suggests refactoring (`suggest-refactor` — a new, real prompt template).
6. Proposes a patch (`propose-patch` — reused `propose_patch()`, now also checked by
   `eng_patch_guard.py`'s two new validations before it's ever shown as applicable).
7. Requires approval before writing files (reused `ApprovalGate`, now actually wired through
   `ToolExecutor` for `apply_patch`, not just `WorkflowExecutor`'s node-level gate).
8. Runs tests only after approval (reused `run_tests()`, `dangerous=True`, approval-gated,
   uniformly time-bounded via `with_timeout()`).

Every tool call is logged to a real `AuditLog` (Module 14) via `ToolExecutor` — not just the
`apply`/`run_tests` nodes `WorkflowExecutor` already gated, but every capability this CLI exposes.

## How success is measured

Curriculum's own six named failure cases, each proven caught for real against a scripted
adversarial response — not asserted, demonstrated:

| Failure case | How Project 3 proves it's caught | Honest-skip status |
|---|---|---|
| Model invents a file path | `resolve_within_sandbox()` (Module 14, reused) raises `PathTraversalError`/`FileNotFoundError` before any read/write | Real |
| Model changes unrelated files | `eng_patch_guard.py`'s new `validate_patch_scope()` rejects a patch whose target file doesn't match the request's expected file | Real |
| Model suggests an unsafe shell command | `eng_command_safety.py`'s new allowlist check rejects any command outside the fixed pytest-invocation shape | Real |
| Model generates an invalid patch | `patch_tools.py`'s `validate_patch_format()`/`apply_patch()` (reused, Module 17) reject malformed or context-mismatched patches | Real |
| Model misses a dependency/import | `run_tests()` (reused) surfaces the real `ImportError`/`NameError` in real pytest output — proven by a scripted patch that omits a needed import, run for real against `demo_repo/` | Real |
| Model creates tests that don't run | Same `run_tests()` path — a scripted "generated test" with a syntax error is run for real and its real failure captured, not assumed | Real |
| Intent classification accuracy | `eng_intent_classifier.py` scored against a real labeled set in `evals/eng_golden_set.jsonl` | Real |
| Patch application correctness (happy path) | A real bug in `demo_repo/`, patched and verified via a real before/after `pytest` run (Module 17's own proof pattern, applied to a new scenario) | Real |

A metric only counts as "measured" in REPORT.md if it has a real, printed result from a real run
of `evals/run_eng_eval.py` — not a claim. Real code-generation *quality* (does a real model
propose a good patch) stays honest-skip, deferred to the resourced 32GB Mac, same as Module 17.
