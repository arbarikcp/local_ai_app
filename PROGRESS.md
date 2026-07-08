# Course build progress

Tracks what has been built against the curriculum in [curriculum.md](curriculum.md) (the bible). See [README.md](README.md) for how to actually read/run each module. Update this file every time a module, project, or infra piece is completed or started.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

## Phase 0 — Repo infrastructure

- [x] Monorepo directory structure created (`docs/`, `models/`, `notebooks/`, `packages/`, `projects/`, `datasets/`, `evals/`, `scripts/`, `reports/`, `docker/`)
- [x] `uv` project initialized (`pyproject.toml`, `.python-version`, `uv.lock`)
- [x] `Makefile` with `sync`/`test`/`lint`/`fmt`/`notebook` targets
- [x] `PROGRESS.md` tracker (this file)
- [x] `models/MODEL_CATALOG.md` (populated in Module 3)
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

## Module 2 detail (done 2026-07-08)

**Hard constraint for this whole repo, discovered/confirmed this module:** the Mac used to
build this course has limited disk/memory and **must never have a model runtime (Ollama,
llama.cpp/llama-cpp-python, MLX) or model weights installed on it.** All model-execution labs
across every module are built to run correctly elsewhere and are deliberately left in a
verified "pending live run on a resourced machine" state here — this is now a standing rule,
not a one-time gap (see the machine-constraints project memory).

Built:
- `docs/modules/02_mac_local_ai_development_environment.md` — theory chapter covering Apple
  Silicon vs Intel, dev tools, Homebrew, uv project setup, all three runtime install paths
  (Ollama, llama.cpp+Metal, llama-cpp-python, MLX), model cache locations per runtime, and
  disk usage/cleanup commands.
- `notebooks/02_mac_environment_setup.ipynb` — executed end-to-end; runs the real dev-tool
  check, cache scan, and all three runtime smoke tests live.
- `scripts/module_02/`: `mac_environment_check.py` (read-only tool/version probe),
  `model_cache.py` (cache location + size scanner, symlink-safe), `smoke_test_ollama.py`,
  `smoke_test_llamacpp_server.py`, `smoke_test_mlx.py`, `smoke_test_runtimes.py` (orchestrator
  that renders the combined deliverable report), `setup_mac.sh` (documented install commands,
  reviewed but **not executed** on this machine). `tests/` adds 23 new pytest unit tests (65
  total in the repo now, all passing); `ruff check .` clean.
- `reports/module_02_environment_report.md` — deliverable. Lab 2.1 (dev tools) fully run and
  found a real gap (`ripgrep`/`rg` binary not actually installed, only shadowed by a shell
  function in interactive terminals — `shutil.which` correctly caught this). Labs 2.2–2.4
  (actual runtime smoke tests) executed and correctly produced skip results with exact
  install commands, since no runtime is installed here by design.

Deliberately not done in Module 2:
- No brew install of Ollama/llama.cpp/MLX and no model download — machine constraint, not an
  oversight. `scripts/module_02/setup_mac.sh` documents the commands for the resourced Mac.
- No `LLMRuntime` abstraction — still Module 6's job.

## Module 3 detail (done 2026-07-08)

Built:
- `models/MODEL_CATALOG.md` — 13 candidate models (chat/code/embedding/reranker categories)
  using curriculum's YAML schema (§6.4), each tagged `verification_status: documented`
  (public model-card info only, none benchmarked here) with per-entry license caveats.
- `docs/modules/03_local_model_selection_and_benchmarking.md` — theory chapter: model cards,
  license checks, base-vs-instruct, chat templates, context length claims, quantized
  variants, benchmark dimensions table, human-vs-automated eval, regression datasets.
- `evals/golden_sets/{summarization,extraction,classification,code,rag,tool_calling}.jsonl`
  — 36 frozen golden records total, 6 task types (exceeds the assessment's 5-task minimum).
- `scripts/module_03/scorers/`: `exact_match.py`, `json_validity.py` (handles the
  markdown-fence-wrapping failure mode from Module 1 §11), `rag_metrics.py` (simplified
  precursor to Module 13's full RAG eval), `rubric_judge.py` (LLM-as-judge with an injected
  judge function, built and tested but not yet wired into any golden set's scorer dispatch).
- `scripts/module_03/run_benchmark.py` — orchestrator: loads a golden set, builds the
  task-appropriate prompt, calls an injected `generate_fn`, scores via the right scorer, and
  renders comparison/scorecard markdown tables. Real usage plugs in
  `default_generate_fn` (Ollama via Module 1's `ollama_probe.py`); tests inject fakes.
- `notebooks/03_model_benchmarking.ipynb` — **executed end-to-end**; ran the full harness
  against two deliberately-imperfect fake models and got genuinely discriminating scores
  (0.00-1.00 spread across tasks, not a rubber stamp), then correctly skipped the
  real-Ollama section on this machine.
- `reports/model_scorecard_TEMPLATE.md` (reusable blank template) and
  `reports/module_03_local_model_selection_report.md` (this module's own deliverable,
  including the fake-model proof table and the exact command to complete the real
  3-model comparison on a resourced Mac).
- 72 new unit tests (114 total in the repo now, all passing); `ruff check .` clean.

Deliberately not done in Module 3:
- No real 3-models-×-5-tasks benchmark run — machine constraint. Harness is fully built and
  proven against fakes; `reports/module_03_local_model_selection_report.md` has the exact
  command and says explicitly what's pending.
- `rubric_judge.py` not wired into any golden set yet — none of the 6 task types currently
  need an LLM-as-judge; will be used once an open-ended task is added.
- No latency/memory/TTFT threading into scorecards yet — `ollama_probe.py` already captures
  those; `run_benchmark.py` doesn't surface them in its tables yet. Noted as a gap, not
  silently dropped.

## Phase 1 — Foundation (Modules 1–6)

| Module | Theory doc | Notebook | Code + tests | Deliverable report | Status |
|---|---|---|---|---|---|
| 1. Local LLM systems thinking | [x] | [x] | [x] | [~] | infra done; empirical labs pending a resourced Mac |
| 2. Mac local AI dev environment | [x] | [x] | [x] | [~] | Lab 2.1 (dev tools) fully done; Labs 2.2-2.4 pending a resourced Mac |
| 3. Local model selection and benchmarking | [x] | [x] | [x] | [~] | harness fully built + proven against fakes; real 3-model run pending a resourced Mac |
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
- **`ollama` is NOT installed on this machine, and never will be — standing constraint, confirmed by the user in Module 2.** This machine has limited disk/memory and is not used to run local models; all course content is built and practiced here, then executed on a separate, better-resourced Mac. Module 1/2 labs that require running an actual local model (multi-model comparison, long-prompt stress test, runtime smoke tests) are written to run correctly there; deliverable reports honestly record "not run — no local runtime available" rather than fabricated numbers, per the course's own honesty rule (§4.1 of the bible: never claim numbers that weren't measured).
- `llama.cpp` / `llama-cpp-python` / `MLX` confirmed not installed (Module 2) — and per the constraint above, will not be installed on this machine.
- Real gap found in Module 2: `ripgrep` (the `rg` binary) is not actually installed here, only shadowed by a terminal shell function — `brew install ripgrep` needed on a fresh machine following this README.

## Working conventions for this build

- Each module gets: `docs/modules/NN_name.md` (theory), `notebooks/NN_name.ipynb` (explanatory + runnable), code under `packages/*` or `scripts/module_NN/` (only once the module's curriculum section says code belongs there — e.g. Module 1 is pre-abstraction and only produces lab scripts + a report, the reusable `LLMRuntime` interface is not introduced until Module 6), unit tests alongside code, and a report in `reports/`.
- Every python file has corresponding pytest unit tests (per user's global coding guideline), run via `make test` before a module is marked done here.
- Module boundaries are respected: a module's code stays inside its own package/script directory; if a change would require touching another module's code, that will be flagged to the user before doing it.
