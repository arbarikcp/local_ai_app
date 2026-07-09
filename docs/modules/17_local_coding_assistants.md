# Module 17 — Local Coding Assistants

> Phase: Agents/tools · Bible reference: [curriculum.md §27](../../curriculum.md#27-module-17--local-coding-assistants)

## Goal

Build a local, repo-aware coding assistant — the capstone of the Agents/tools phase, wiring
together everything Modules 14-16 built.

```text
User request
  -> classify coding task
  -> repo index search
  -> read relevant files
  -> build context
  -> generate answer or patch
  -> validate patch format
  -> optional human approval
  -> apply patch
  -> run tests
  -> report result
```

> **Machine note:** every stage of this pipeline runs for real except the one LLM call that
> proposes a patch or answer (`FakeRuntime`-backed, real adapter unchanged later). Symbol
> parsing, patch format validation, patch application, and test execution are all real,
> deterministic code with no honest-skip surface.

## The sample repo: `datasets/code_repos/mini_calculator/`

A tiny, real, committed Python package with **a genuine pre-existing bug and a genuine failing
test** - `average([])` raises `ZeroDivisionError` instead of returning `0.0`, and
`test_average_of_empty_list_should_return_zero` fails against the unpatched code. This is not a
staged example computed after the fact: `scripts/module_17/` labs run the real test suite
against the real (buggy) code first, observe a real failure, propose and apply a real patch, and
run the real test suite again to observe a real pass — the strongest form of "real proof" this
course has used, extending Module 15's "before/after" checkpoint demonstration to an actual code
fix.

## Repo structure note

New tools join Module 14's `packages/local_ai_agents/tools/` (same reasoning as Module 16:
these are tools, they belong with the tools package, not a new top-level package).
`coding_assistant.py` lives at `packages/local_ai_agents/` top level, mirroring `pipeline.py`'s
role in `local_ai_rag` (Module 11) - the orchestration layer that wires primitives from several
subpackages into one real pipeline matching curriculum's architecture diagram.

## Core topics

### 1. Code model selection

Not implemented as code - same discipline as Module 3's general model selection, extended:
curriculum's own guidance (a strong code-specialized local model, benchmarked not assumed) is
documented, not re-implemented; Module 3's benchmarking harness already generalizes to this.

### 2. Code chunking

Reused from Module 12's `structural_chunker.py` concept (never split an atomic block) applied
conceptually to code - but this module's actual context-building step (`read_file`) reads whole
functions by line range located via `list_symbols`, not chunked embeddings; embedding-based code
search is explicitly out of scope (see §6).

### 3. AST-aware parsing

`tools/list_symbols.py`'s `list_symbols()` - real Python `ast` module parsing, not a
regex-based guess at function/class boundaries. Walks the parsed tree for
`FunctionDef`/`AsyncFunctionDef`/`ClassDef` nodes and their real line numbers.

### 4. Symbol search

The same `list_symbols()` output, filtered by name - "where is `average` defined" is answered by
real AST data, not a text search that might match a comment or a string literal containing the
word "average".

### 5. Repo map

`scripts/module_17/index_repo_demo.py`'s repo map is the real output of `list_symbols()` across
every `.py` file in the sample repo - a real structural index, not a hand-written description of
the repo.

### 6. Code embeddings

**Not implemented this module.** Module 9's `Embedder`/`FakeEmbedder` infrastructure already
exists and would work unchanged over code text; this module's repo is small enough (2 files)
that lexical `search_repo` (§7) is sufficient to prove the architecture, and adding embedding
based semantic code search here would be repeating Module 9-11's already-proven pattern rather
than testing anything new.

### 7. Hybrid code search

**Not implemented this module**, for the same reason as §6 - Module 10's `hybrid_search()`
already exists and generalizes to code text without new code; a 2-file sample repo has no
retrieval-quality signal to measure that would justify wiring it in here.

### 8. Test generation

`scripts/module_17/generate_tests_demo.py` - a real LLM call (`FakeRuntime`-backed) proposing a
test function's source, written to the sample repo's test file (sandboxed, approval-gated same
as any write), then **actually executed** via `run_tests.py` to confirm it's syntactically valid
and runs - not just that the model produced plausible-looking text.

### 9. Patch proposal

`tools/patch_tools.py`'s `propose_patch()` - one LLM call given the instruction and real file
contents, asked to respond with a unified diff. Mechanically real, `FakeRuntime`-backed;
real-model patch *quality* is deferred to the resourced Mac.

### 10. Human approval

Reused unchanged from Module 14/15's `ApprovalGate` - `apply_patch` and `run_tests` are both
gated, curriculum's own "with approval" annotation on both tools in the required-tools list.

### 11. Safe file writes

`apply_patch()` never touches a file outside `sandbox.py`'s containment, and never applies a
hunk whose context doesn't match the file's actual current content (`PatchFormatError` on
mismatch) - a patch that doesn't cleanly apply is rejected, not partially applied.

### 12. Running tests

`tools/run_tests.py`'s `run_tests()` - a real `subprocess` call to `pytest` inside the sandboxed
repo directory, with a real timeout (reusing the safety-budget discipline from Module 15) and
real captured stdout/exit code - not a simulated pass/fail.

### 13. Code hallucination

Made measurable, not just named: `validate_patch_format()` rejects a syntactically invalid diff
before it's ever applied, and `apply_patch()`'s context-matching check catches a patch whose
hunk describes code that doesn't actually exist in the file (a hallucinated line, a wrong line
number, a file that's drifted since the model last saw it) - §"Real proof" in the deliverable
report demonstrates a rejected hallucinated patch alongside the real accepted one.

## Required tools

| Tool | File | Real mechanism |
|---|---|---|
| `search_repo(query)` | `tools/search_repo.py` | Real lexical search across `.py` files, with matched line numbers |
| `read_file(path, start_line, end_line)` | `tools/read_file.py` | Real sandboxed line-range read |
| `list_symbols(path)` | `tools/list_symbols.py` | Real AST parsing |
| `propose_patch(files, instruction)` | `tools/patch_tools.py` | Real LLM call (`FakeRuntime`-backed), real prompt assembly from real file contents |
| `apply_patch(patch)` with approval | `tools/patch_tools.py` | Real unified-diff parsing/application, context-checked, `ApprovalGate`-gated |
| `run_tests(command)` with approval or sandbox | `tools/run_tests.py` | Real `subprocess` execution, sandboxed cwd, real timeout, `ApprovalGate`-gated |

## Hands-on labs

1. **Index a small Python repo** — `scripts/module_17/index_repo_demo.py`, real `list_symbols()`
   + `search_repo()` over `datasets/code_repos/mini_calculator/`.
2. **Ask architecture questions** — same script: "what functions exist in `calculator.py`?"
   answered from real AST data.
3. **Generate tests for one function** — `scripts/module_17/generate_tests_demo.py`.
4. **Propose a patch** — `scripts/module_17/patch_and_test_demo.py`, fixing the real
   `average([])` bug.
5. **Validate patch** — same script, including a deliberately hallucinated patch rejected.
6. **Run tests** — same script: real failure before the patch, real pass after.
7. **Add human approval** — same script: `apply_patch`/`run_tests` both denied by default,
   approved via a real gate.

## Deliverable

```text
datasets/code_repos/mini_calculator/
packages/local_ai_agents/
  tools/
    read_file.py
    list_symbols.py
    search_repo.py
    patch_tools.py
    run_tests.py
    tests/
  coding_assistant.py
  tests/
scripts/module_17/
  index_repo_demo.py
  generate_tests_demo.py
  patch_and_test_demo.py
reports/module_17_coding_assistant_report.md
```
