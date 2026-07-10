# Project 3 — Local Engineering Assistant

> See [PROPOSAL.md](PROPOSAL.md) for why, [ARCHITECTURE.md](ARCHITECTURE.md) for how it's built,
> [REPORT.md](REPORT.md) for measured results, [OUTRO.md](OUTRO.md) for what's next.

## Setup

No install beyond the repo's own `uv sync` (run once at the repo root) — this project adds no
new dependencies.

## Run the tests

```bash
uv run pytest projects/03_engineering_assistant -q
```

`demo_repo/` is deliberately excluded from this run (`pyproject.toml`'s `norecursedirs`) — it
contains one real, currently-failing test proving a real bug (see below). Run it directly to see
that for yourself:

```bash
cd projects/03_engineering_assistant/demo_repo && uv run python -m pytest tests -q
# 1 failed, 10 passed
```

## Run the CLI

```bash
uv run python projects/03_engineering_assistant/app/eng_cli.py --help
```

⚠️ Every command defaults to `--repo-dir demo_repo` (the committed fixture) and
`--config-path config/app.example.yaml` (which writes to the real `~/.local-llm-ai` on your
machine — same disclosed behavior as Modules 23 and Projects 1-2). **`apply-patch` and
`run-tests` will modify whatever `--repo-dir` you point them at.** To experiment safely, copy
`demo_repo/` somewhere else first:

```bash
cp -r projects/03_engineering_assistant/demo_repo /tmp/demo_repo_sandbox
```

### Example commands

```bash
# Read-only, safe to run against the real demo_repo/ directly:
uv run python projects/03_engineering_assistant/app/eng_cli.py explain-repo
uv run python projects/03_engineering_assistant/app/eng_cli.py search quantity
uv run python projects/03_engineering_assistant/app/eng_cli.py explain-symbol remove_stock
uv run python projects/03_engineering_assistant/app/eng_cli.py generate-tests calculate_discount
uv run python projects/03_engineering_assistant/app/eng_cli.py suggest-refactor inventory/reports.py

# Write operations - use --repo-dir pointing at a sandbox copy, and --approve:
uv run python projects/03_engineering_assistant/app/eng_cli.py propose-patch \
  "fix remove_stock to reject removing more than available" inventory/stock.py \
  --repo-dir /tmp/demo_repo_sandbox

uv run python projects/03_engineering_assistant/app/eng_cli.py apply-patch \
  fix.patch inventory/stock.py --repo-dir /tmp/demo_repo_sandbox --approve

uv run python projects/03_engineering_assistant/app/eng_cli.py run-tests \
  --repo-dir /tmp/demo_repo_sandbox --approve
```

Without `--approve`, `apply-patch` and `run-tests` are denied by the default fail-closed
`NullApprovalGate` (Module 14) — this is intentional, not a bug.

The default runtime is `FakeRuntime` (no model runtime installed on this dev machine) —
`propose-patch`/`explain-symbol`/`generate-tests`/`suggest-refactor` will all return
`FakeRuntime`'s canned response until a real runtime is injected (see ARCHITECTURE.md's
composition root — swap via `build_eng_context(..., runtime=...)`, no other code change needed).

## Run the evaluation

```bash
uv run python projects/03_engineering_assistant/evals/run_eng_eval.py
```

Runs two real harnesses against real, temporary sandboxed copies of `demo_repo/` (never the
committed fixture): intent classification accuracy against
`evals/eng_golden_set.jsonl` (16 labeled requests), and curriculum's own six named failure
cases, each proven caught for real. See REPORT.md for the actual numbers from a real run.

## Available CLI commands

| Command | Requires `--approve` | Purpose |
|---|---|---|
| `explain-repo` | No | List every function/class symbol across the repo (real AST parsing) |
| `search <query>` | No | Search the repo for a query string, real line numbers |
| `explain-symbol <name>` | No | Explain a function/class (real AST location + a real LLM call) |
| `generate-tests <name>` | No | Generate a pytest test function for a symbol |
| `suggest-refactor <file>` | No | Suggest improvements for a whole file |
| `propose-patch <instruction> <file>` | No | Propose a unified-diff patch (doesn't write anything) |
| `apply-patch <patch-file> <file>` | **Yes** | Apply a validated patch to a sandboxed file |
| `run-tests` | **Yes** | Run the repo's pytest suite in a sandboxed subprocess |
