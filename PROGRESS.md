# Course build progress

Tracks what has been built against the curriculum in [curriculum.md](curriculum.md) (the bible). See [README.md](README.md) for how to actually read/run each module. Update this file every time a module, project, or infra piece is completed or started.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

## Phase 0 — Repo infrastructure

- [x] Monorepo directory structure created (`docs/`, `models/`, `notebooks/`, `packages/`, `projects/`, `datasets/`, `evals/`, `scripts/`, `reports/`, `docker/`)
- [x] `uv` project initialized (`pyproject.toml`, `.python-version`, `uv.lock`)
- [x] `Makefile` with `sync`/`test`/`lint`/`fmt`/`notebook` targets
- [x] `PROGRESS.md` tracker (this file)
- [ ] `models/MODEL_CATALOG.md`
- [ ] `docs/architecture.md`, `docs/glossary.md`

## Module 1 detail (done 2026-07-08)

Built:
- `docs/modules/01_local_llm_systems_thinking.md` — full theory chapter (13 sections:
  operational definition, hosted-vs-local comparison, weights/activations/KV cache, unified
  memory, tokenization + the tiktoken warning, context window, prompt/generated token split,
  TTFT/TPS, latency vs throughput, quantization preview, small-model failure taxonomy, why
  RAG matters more for small models, why local ≠ secure).
- `notebooks/01_local_llm_basics.ipynb` — executed end-to-end (`jupyter nbconvert --execute`);
  computes real weight/KV-cache numbers for a 7B-class model shape, demonstrates the
  heuristic-vs-exact tokenizer distinction, and drives the lab scripts with honest
  skip-if-unavailable behavior.
- `scripts/module_01/`: `ollama_probe.py` (lab-local HTTP probe — explicitly NOT the
  canonical `LLMRuntime`, which is Module 6's job), `token_estimate.py` (heuristic + exact
  HF-tokenizer counting), `lab_1_1_multi_model_run.py`, `lab_1_2_long_prompt_stress_test.py`,
  `lab_1_3_small_model_failure_analysis.py`, plus `tests/` with 19 pytest unit tests (all
  passing) covering every pure-logic path (dataclass property derivations, prompt-length
  math, markdown report formatting). `ruff check .` clean.
- `reports/module_01_local_llm_observations.md` — deliverable report. Memory-math predictions
  filled in from real computed values; Labs 1.1–1.3 empirical results left explicitly
  "pending live run" with exact commands to complete them, since this machine has no local
  model runtime installed yet (that's Module 2's job).

Deliberately not done in Module 1 (belongs to a later module):
- No `LLMRuntime` abstraction/package code under `packages/local_ai_core/` — that's Module 6.
- No actual model install/run — that's Module 2 (environment setup) prerequisite work.

## Phase 1 — Foundation (Modules 1–6)

| Module | Theory doc | Notebook | Code + tests | Deliverable report | Status |
|---|---|---|---|---|---|
| 1. Local LLM systems thinking | [x] | [x] | [x] | [~] | infra done; empirical labs blocked on Ollama install (Module 2) |
| 2. Mac local AI dev environment | [ ] | [ ] | [ ] | [ ] | not started |
| 3. Local model selection and benchmarking | [ ] | [ ] | [ ] | [ ] | not started |
| 4. Quantization, context, memory math | [ ] | [ ] | [ ] | [ ] | not started |
| 5. Serving local models | [ ] | [ ] | [ ] | [ ] | not started |
| 6. Python client architecture | [ ] | [ ] | [ ] | [ ] | not started |

## Phase 1.5 — Serving/performance foundation

| Module | Status |
|---|---|
| 6.5 Serving concurrency, batching, caching | not started |
| 20. Inference optimization under 8–24GB | not started |

## Phase 2 — Application primitives (Modules 7–10, 8.5)

All not started.

## Phase 3 — RAG (Modules 11–13)

All not started.

## Phase 4 — Agents/tools (Modules 14–17)

All not started.

## Phase 5 — Advanced (Modules 18–19)

All not started.

## Phase 6 — Production (Modules 21–23)

All not started.

## Projects & capstone

All not started.

---

## Environment notes (this machine)

Captured during Phase 0 setup, `2026-07-08`:

- Python 3.13.5 available system-wide; project pinned to Python 3.12 via `uv`.
- `uv` 0.6.8 available.
- **`ollama` is NOT installed on this machine.** Module 1/2 labs that require running an actual local model (multi-model comparison, long-prompt stress test, runtime smoke tests) cannot be executed end-to-end here yet. Code and labs are written to run correctly once a runtime is installed; until then, deliverable reports honestly record "not run — no local runtime available" rather than fabricated numbers, per the course's own honesty rule (§4.1 of the bible: never claim numbers that weren't measured).
- `llama.cpp` / `llama-cpp-python` / `MLX` also not verified installed — to be checked in Module 2.

## Working conventions for this build

- Each module gets: `docs/modules/NN_name.md` (theory), `notebooks/NN_name.ipynb` (explanatory + runnable), code under `packages/*` or `scripts/module_NN/` (only once the module's curriculum section says code belongs there — e.g. Module 1 is pre-abstraction and only produces lab scripts + a report, the reusable `LLMRuntime` interface is not introduced until Module 6), unit tests alongside code, and a report in `reports/`.
- Every python file has corresponding pytest unit tests (per user's global coding guideline), run via `make test` before a module is marked done here.
- Module boundaries are respected: a module's code stays inside its own package/script directory; if a change would require touching another module's code, that will be flagged to the user before doing it.
