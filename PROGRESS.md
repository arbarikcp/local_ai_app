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

## Module 4 detail (done 2026-07-08)

Built:
- `docs/modules/04_quantization_context_and_memory_math.md` — theory chapter: quantization
  formats and GGUF naming, quality/performance trade-offs, the exact weights and KV-cache
  formulas (extending Module 1's preview), the full memory budget, prompt compression,
  batch/concurrency, Apple unified memory, runtime overhead, KV-cache quantization as a
  first-class lever, and reranker/embedder memory contention.
- `scripts/module_04/memory_math.py` — `weights_bytes`, `kv_cache_bytes`,
  `estimate_memory_budget`, unit-tested against every number in the theory doc's tables.
  Documents and preserves the curriculum's own mixed-unit convention (decimal GB for
  weights, binary GiB for KV cache) rather than silently "fixing" it.
- `scripts/module_04/model_shapes.py` — documented (not measured) architecture shapes for
  4 course models (Llama 3.1 8B, Qwen2.5 7B/1.5B, Qwen2.5-Coder 7B).
- `scripts/module_04/memory_sampler.py` — **real, working** process-RSS peak sampler
  (`ps`-based) and process-finder (`pgrep`-based), proven in the executed notebook against a
  dummy subprocess that allocates ~200MB — correctly tracked a 148MB+ peak.
- `scripts/module_04/lab_4_{1,2,3,4}_*.py` — all four labs (quantization comparison, context
  scaling, concurrency simulation, predict-then-measure), each wired to Module 1's
  `ollama_probe.py` and this module's memory sampler, each with the same honest-skip
  behavior as every prior module's labs.
- `notebooks/04_quantization_context_memory_math.ipynb` — **executed end-to-end**;
  reproduces every theory-doc worked example as live computed numbers, proves the memory
  sampler works, and correctly skips the real-Ollama section.
- `reports/module_04_quantization_context_memory_report.md` — deliverable. Also caught and
  documented one real discrepancy: the theory doc's 128K-context KV-cache row is a rounded
  approximation (~16.0 GiB) of a non-power-of-two token count; the exact formula computes
  15.625 GiB for literally 128,000 tokens — noted rather than silently forced to match.
- 57 new unit tests (171 total in the repo now, all passing); `ruff check .` clean.

Deliberately not done in Module 4:
- No real quantization comparison, context-scaling measurement, concurrency simulation, or
  predict-vs-actual gap analysis — machine constraint. All four labs are built, unit-tested,
  and given exact commands to run on a resourced Mac in the deliverable report.
- `lab_4_3`'s `ConcurrencyLevelResult` reports `failure_rate`, not the curriculum's literal
  `timeout_rate` — Module 1's `ollama_probe.generate` can't yet distinguish a timeout from
  any other httpx error; a real `RequestTimeout` error type is Module 6's job. Documented in
  the lab's own docstring rather than reporting a falsely-precise metric name.

## Module 5 detail (done 2026-07-08)

**Target execution hardware confirmed this module: a 32GB Mac.** Above all three course RAM
tiers (8/16/24GB) — worth revisiting the model catalog (Module 3) toward 14B-class models
once real benchmarking starts, not staying capped at what an 8-24GB machine could do.

Built:
- `docs/modules/05_serving_local_models.md` — theory chapter: direct CLI vs. local HTTP API
  vs. OpenAI-compatible APIs, streaming (NDJSON for Ollama, SSE for OpenAI-compatible
  servers), runtime lifecycle/warmup/unloading, prompt caching, request cancellation (with
  an explicit verification-limits caveat), error handling, and the three serving patterns
  (direct/gateway/router).
- `scripts/module_05/serve_ollama.sh`, `serve_llamacpp.sh` — idempotent, health-checked
  repeatable start scripts (`bash -n` syntax-checked, reviewed, not executed — would start a
  runtime).
- `scripts/module_05/ollama_streaming.py` — NDJSON stream parsing, chunk accumulation, real
  (not approximated) TTFT from first streamed chunk, tokens/sec, and a cancellation demo with
  an honestly-documented verification limit (client-side elapsed time is a proxy, not proof
  of server-side compute cancellation).
- `scripts/module_05/ollama_metadata.py` — `/api/show` probe and parser, including a dynamic
  `<family>.context_length` key lookup verified against a realistic fixture.
- `scripts/module_05/warmup_experiment.py` — cold-vs-warm TTFT orchestration and statistics,
  injected TTFT function for full testability.
- `scripts/module_05/feature_matrix.py` — 4-runtime × 6-feature comparison table
  (structured output, grammar, token counting, streaming, cancellation, usage reporting),
  every entry tagged `documented`/`verified=False` pending real measurement.
- `scripts/module_05/llamacpp_openai_streaming.py` — OpenAI-client streaming against a local
  server, extending Module 2's non-streaming smoke test.
- `scripts/module_05/run_mlx_generate.py` — MLX warmup + streaming demo, reusing Module 2's
  Apple-Silicon/importability checks rather than duplicating them.
- `notebooks/05_serving_local_models.ipynb` — **executed end-to-end**; parsers proven against
  realistic fixtures, feature matrix rendered, warmup statistics proven with a fake TTFT
  function, real-runtime cells correctly skip (confirmed this machine **is** Apple Silicon
  but has no `mlx_lm` installed, so the MLX skip message is accurate, not a false negative).
- `reports/module_05_runtime_serving_matrix.md` — deliverable, including the feature matrix
  and a full write-up of a real bug this module caught (see below).
- 59 new unit tests (230 total in the repo now, all passing); `ruff check .` clean.

**Real bug caught and fixed during this module's own build:** `run_mlx_generate.py`'s
`summary_to_markdown` had adjacent f-string literals followed by an `if/else` — Python
concatenates adjacent string literals *before* applying a trailing ternary, so the whole
multi-line report would have silently collapsed to one line whenever `stream_total_seconds`
was `None`. Caught by writing the test for that case before considering the function done,
not by inspection. Fixed and left the regression test in place with an explanatory comment.

Deliberately not done in Module 5:
- No real per-runtime measurement (flipping `feature_matrix.py` entries from `documented` to
  `measured`) — machine constraint. Every parser/orchestrator that will produce real
  observations is built and unit-tested; completing this is running it, not building it.
- The `failure_rate`-vs-`timeout_rate` gap noted in Module 4 is still open and now also
  affects `ollama_streaming.py`'s cancellation demo — still deferred to Module 6's error
  taxonomy, not patched ad hoc per-module.

## Module 6 detail (done 2026-07-08)

**The canonical `LLMRuntime` abstraction now exists** — `packages/local_ai_core/runtimes/`.
Every module from here on uses this instead of lab-local code. Unlike every prior module,
this one needed no honest-skip labs: `FakeRuntime` and `httpx.MockTransport` let the whole
abstraction be built and verified without a live runtime.

Built:
- `docs/modules/06_python_client_architecture.md` — theory chapter covering the runtime
  abstraction, request/response types, streaming interface, the error taxonomy (and how it
  resolves Modules 4/5's `failure_rate`-vs-`timeout_rate` gap), retries, timeouts, metrics
  hooks, dependency injection, and the two-tier testing strategy (`FakeRuntime` +
  `httpx.MockTransport`) with an explicit statement of what that does/doesn't prove.
- `packages/local_ai_core/runtimes/types.py` — `LLMRequest`/`LLMResponse`/`ResponseFormat`
  Pydantic models, matching curriculum.md §16 exactly.
- `errors.py` — the full 10-member `LLMError` taxonomy.
- `base.py` — the `LLMRuntime` Protocol, `ensure_trace_id`, `MetricsHook`/
  `NullMetricsHook`/`LoggingMetricsHook`, `Timer`, and `with_retries` (exponential backoff,
  retryable-vs-not selection).
- `fake.py` — `FakeRuntime`: canned/per-model responses, `fail_with` and
  `fail_first_n_calls` failure injection (built specifically to test retry logic
  deterministically), streaming, tokenize.
- `ollama.py` — real adapter with precise httpx-exception-to-taxonomy mapping (connect vs.
  read vs. pool timeout, finally resolving the gap flagged in Modules 4 and 5); `tokenize()`
  correctly raises `FeatureNotSupported` rather than faking a count, pointing callers at
  Module 1's `HFTokenizerCounter` instead.
- `openai_compatible.py` — real adapter using the `openai` SDK for `generate`/`stream`, plus
  a raw-httpx call to llama.cpp's native `/tokenize` endpoint.
- `mlx.py` — real adapter bridging `mlx_lm`'s synchronous `stream_generate` to a genuine
  async generator via a background thread + queue, with model-load caching.
- `tests/test_runtime_contract.py` — the curriculum's explicit ask: one shared suite proving
  all four adapters are interchangeable.
- `notebooks/06_python_client_architecture.ipynb` — **executed end-to-end**, live-demonstrating
  every adapter (not honest-skip stubs) since none of this module's core logic needs a real
  runtime to verify.
- `reports/module_06_python_client_architecture_report.md` — deliverable, including full
  write-ups of two real bugs this module's own tests caught (below).
- 167 new tests (165 passing + 2 correctly-skipped; 395 total in the repo, all passing);
  `ruff check .` clean.

**Two real bugs caught by this module's own tests:**
1. The `openai` SDK retries transient errors internally by default; composed with this
   module's own `with_retries()`, that would silently multiply retry attempts. Noticed
   because the adapter's test file took an anomalous 3.01s (vs. ~0.1-0.3s for siblings) —
   fixed by setting `max_retries=0` on every `AsyncOpenAI` client this module constructs, so
   retry policy lives in exactly one place. Test runtime dropped to 0.72s.
2. `test_runtime_contract.py`'s first version wrongly assumed every adapter rejects
   `response_format.type="grammar"`. It failed immediately for `OpenAICompatibleRuntime` —
   correctly, since real llama.cpp servers DO support grammar (per Module 5's
   `feature_matrix.py`). Fixed by making per-adapter capability expectations explicit
   instead of assumed-uniform.

Deliberately not done in Module 6:
- No real-server verification that `httpx.MockTransport`'s assumed response shapes match an
  actual running Ollama/llama.cpp server byte-for-byte — machine constraint, narrower here
  than prior modules since the adapter logic itself is fully verified; see the report's
  "What this module's testing does and does not prove."
- `ToolCallValidationError` and `SafetyPolicyViolation` are declared in the taxonomy but not
  yet raised by any adapter — intentionally: they're Module 14's and Module 22's territory
  respectively, declared now so the taxonomy is complete rather than extended piecemeal later.

## Module 6.5 detail (done 2026-07-08)

Like Module 6, most of this module needed no honest-skip labs: `FakeRuntime`'s simulated
latency (Module 6) is exactly what queueing/caching proofs need.

Built:
- `docs/modules/06_5_serving_concurrency_batching_caching.md` — theory chapter: local vs.
  cloud serving, single- vs. multi-sequence behavior, queueing vs. rejection, Ollama/llama.cpp
  concurrency knobs, the context-per-slot trap, the full caching strategy table, the
  invariant-prompt-prefix-first layout rule, thermal throttling/backpressure, and why
  `max_concurrent_requests: 1` is the honest default.
- `packages/local_ai_core/gateway/queue.py` — `BoundedRequestQueue`: real concurrency
  limiting (measured, not assumed), queue-wait-vs-execution-time split, admission rejection.
- `cache.py` — `ResponseCache` (exact-match, LRU), `SemanticCache` (cosine-similarity,
  conservative default threshold, returns the matched score for auditability),
  `EmbeddingCache` (LRU, ready for Module 9-11's ingestion pipeline). `response_cache_key()`
  takes quantization/tool/schema/safety-policy versions as explicit parameters so omitting
  one is a visible choice.
- `admission_control.py` — `AdmissionPolicy` (rejects unjustified concurrency > 1 at
  construction time), `recommend_policy_from_measurements()` (p95-and-failure-rate-gated).
- `scripts/module_06_5/lab_caching_before_after.py` — **runs for real, no runtime needed**;
  produced a genuine 4.05x speedup and 75% hit rate against `FakeRuntime`'s simulated latency.
- `scripts/module_06_5/lab_measure_concurrency.py` — real 1/2/4-concurrency measurement
  harness using this module's own queue/admission-control layer (extending Module 4's raw
  concurrency simulation, which had neither); honest-skip pending the resourced Mac.
- `notebooks/06_5_serving_concurrency_batching_caching.ipynb` — **executed end-to-end**,
  every piece of infrastructure demonstrated live with real numbers.
- `reports/module_06_5_serving_concurrency_report.md` — deliverable, including a full
  write-up of a real concurrency-accounting bug this module's own tests caught (below).
- 88 new tests (481 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

**A real bug caught by this module's own tests:** `BoundedRequestQueue`'s first
implementation conflated "no room to wait" with "no room to run" — `max_queue_size=0`
rejected even the very first, uncontended request, because the admission check compared a
`_waiting` counter incremented on every submission against `max_queue_size` before checking
whether a concurrency slot was actually free. A second, related bug: a request that had to
wait was never counted as "running" once admitted, so subsequent admission decisions could
undercount real concurrency. Fixed by tracking `_running`/`_waiting` explicitly as the
source of truth for admission, with the semaphore kept strictly in lockstep as the blocking
mechanism rather than the source of truth. Two regression tests guard both bugs.

Deliberately not done in Module 6.5:
- No real 1/2/4-concurrency measurement against an actual runtime, and no comparison of two
  runtime settings (e.g. `OLLAMA_NUM_PARALLEL` values) — machine constraint. Harness fully
  built and unit-tested; completing this is running `lab_measure_concurrency.py` twice.
- KV-prefix reuse (theory doc §9) is documented but not implemented as application code —
  it's runtime-level behavior, not something this layer controls directly.

## Phase 1 — Foundation (Modules 1–6)

| Module | Theory doc | Notebook | Code + tests | Deliverable report | Status |
|---|---|---|---|---|---|
| 1. Local LLM systems thinking | [x] | [x] | [x] | [~] | infra done; empirical labs pending a resourced Mac |
| 2. Mac local AI dev environment | [x] | [x] | [x] | [~] | Lab 2.1 (dev tools) fully done; Labs 2.2-2.4 pending a resourced Mac |
| 3. Local model selection and benchmarking | [x] | [x] | [x] | [~] | harness fully built + proven against fakes; real 3-model run pending a resourced Mac |
| 4. Quantization, context, memory math | [x] | [x] | [x] | [~] | formulas verified against every theory-doc number; real measurement pending a resourced Mac |
| 5. Serving local models | [x] | [x] | [x] | [~] | feature matrix + all parsers built and tested; real per-runtime measurement pending a resourced Mac |
| 6. Python client architecture | [x] | [x] | [x] | [x] | complete — canonical LLMRuntime abstraction built and fully verified via FakeRuntime + httpx.MockTransport, no honest-skip labs needed |

## Phase 1.5 — Serving/performance foundation

| Module | Status |
|---|---|
| 6.5 Serving concurrency, batching, caching | complete — gateway infra (queue/cache/admission control) fully built and verified; real 1/2/4-concurrency measurement pending a resourced Mac |
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
- **Target execution hardware confirmed in Module 5: a 32GB Mac.** This is above all three course RAM tiers (8/16/24GB) — when real benchmarking starts (Module 3 rerun, Module 4 measurement), revisit `models/MODEL_CATALOG.md` to include 14B-class models rather than staying capped at what an 8-24GB machine could do. Chip (Apple Silicon vs. Intel) not yet confirmed — this machine's own `uname -m` is `arm64`/Apple Silicon (Module 2), but that doesn't tell us about the target machine; confirm before assuming MLX labs are runnable there.

## Working conventions for this build

- Each module gets: `docs/modules/NN_name.md` (theory), `notebooks/NN_name.ipynb` (explanatory + runnable), code under `packages/*` or `scripts/module_NN/` (only once the module's curriculum section says code belongs there — e.g. Module 1 is pre-abstraction and only produces lab scripts + a report, the reusable `LLMRuntime` interface is not introduced until Module 6), unit tests alongside code, and a report in `reports/`.
- Every python file has corresponding pytest unit tests (per user's global coding guideline), run via `make test` before a module is marked done here.
- Module boundaries are respected: a module's code stays inside its own package/script directory; if a change would require touching another module's code, that will be flagged to the user before doing it.
