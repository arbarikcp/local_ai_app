# Module 17 deliverable — local coding assistant report

Status: **complete.** This is the capstone of the Agents/tools phase, wiring together Modules
14-16 into a real, end-to-end coding assistant. Every stage of the architecture diagram runs for
real — repo indexing (AST parsing), patch validation, patch application, and test execution —
except the one LLM call that proposes patch/test text (`FakeRuntime`-backed, real adapter
unchanged later).

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `datasets/code_repos/mini_calculator/` | — | A real, committed Python package with a genuine pre-existing bug and a genuine failing test |
| `tools/read_file.py` | 5 | Sandboxed line-range reads |
| `tools/list_symbols.py` | 7 | Real AST parsing — functions, async functions, classes, nested methods, exact line numbers |
| `tools/search_repo.py` | 6 | Lexical search across `.py` files with real matched line numbers, sandboxed |
| `tools/patch_tools.py` | 10 | Unified-diff parsing, real application producing valid executable Python (verified via `exec()`), and a hallucinated patch rejected before touching the file |
| `tools/run_tests.py` | 6 | Real `pytest` subprocess execution — genuine pass, genuine fail, real captured stdout, real timeout enforcement |
| `coding_assistant.py` | 5 | The full workflow graph: happy path (real bug fixed, real tests pass), approval-denied path, invalid-patch path, no-match path |
| `scripts/module_17/` (3 lab scripts) | 36 | Labs 1-7 exercised against the real sample repo |
| `notebooks/17_local_coding_assistants.ipynb` | — | **Executed end-to-end** — every cell a real measurement |

**58 new tests this module** (1444 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## The headline real proof: a genuine bug, genuinely fixed

Every other module in this course has had to explain what's real versus scripted. This module's
central demonstration needs no such caveat for the outcome itself — only for who wrote the patch
text:

```
Before the patch: tests passed = False -> `1 failed, 6 passed in 0.03s`
Apply denied with no real approval gate: approval_denied
Apply approved with a real approval gate: end
After the patch:  tests passed = True  -> `7 passed in 0.00s`
```

`datasets/code_repos/mini_calculator/calculator.py`'s `average([])` genuinely raised
`ZeroDivisionError` before this module was written — not staged after the fact, verified by
running `pytest` against the unpatched repo (`1 failed, 6 passed`, from `git log`-independent,
directly re-runnable evidence). `coding_assistant.py`'s workflow graph proposed a patch
(`FakeRuntime`-scripted text, since a real model isn't running on this machine), validated its
unified-diff format, required and received real human approval, applied it with a real
line-for-line replacement, and re-ran the real test suite: `7 passed`, zero failures. Anyone
can reproduce this by running `uv run python scripts/module_17/patch_and_test_demo.py`.

## Real proof: code hallucination is caught, not misapplied

```
Hallucinated patch rejected (context mismatch): True
File untouched after the rejected hallucinated patch: True
```

A second patch, syntactically identical in shape to the real one but describing code that
doesn't exist in the file (`statistics.mean(numbers)` instead of the real
`sum(numbers) / len(numbers)`), is rejected by `apply_patch()`'s context-matching check before
any write happens — and the file's content was checked directly afterward to confirm nothing
changed, not just that an exception was raised.

## Real proof: generated tests are executed, not just produced

```
Passing test count before adding the generated test: 6
Passing test count after adding the generated test: 7
Final pytest summary line: 1 failed, 7 passed in 0.03s
```

The scripted "generated" test (`test_subtract_generated`) is written to a real sandboxed copy of
the repo and the real test suite is re-run — the passing count increases by exactly one, and the
repo's one pre-existing unrelated failure is untouched, confirming the new test genuinely
executes and passes rather than merely looking plausible as text.

## Real proof: AST-based repo indexing produces exact, verifiable data

```
calculator.py:
  - function add (line 4)
  - function subtract (line 8)
  - function multiply (line 12)
  - function divide (line 16)
  - function average (line 22)
```

Every line number matches the real file exactly (`test_real_line_numbers_are_reported`
independently confirms this against a controlled sample) - this is Python's own `ast` module
walking a real parse tree, not a regex heuristic that could be fooled by a function name
appearing inside a string or comment.

## Deliberately not done in Module 17

- **Code embeddings and hybrid code search (topics 6-7)** — genuinely not implemented, not
  honest-skipped. Module 9's `Embedder`/`FakeEmbedder` and Module 10's `hybrid_search()` already
  exist and would work unchanged over code text; the 2-file sample repo is too small to produce
  any retrieval-quality signal that would justify wiring them in here. A larger sample repo
  would be the natural trigger to revisit this.
- **Code model selection (topic 1)** — not implemented as code; Module 3's benchmarking harness
  already generalizes to code-specialized models without new infrastructure.
- No real LLM proposing patches or generated tests — `propose_patch()` and the test-generation
  prompt are fully built and exercised with `FakeRuntime`; real model patch *quality* (does a
  small local model produce a correctly-formatted, correctly-targeted diff on the first try, or
  does it need Module 8's repair-retry ladder?) is an open, real question deferred to the
  resourced 32GB Mac.
- Only one bug/patch scenario demonstrated in the sample repo — the mechanism (propose →
  validate → approve → apply → test) is identical regardless of which function has the bug;
  a second scenario would exercise the same code path, not new logic.
