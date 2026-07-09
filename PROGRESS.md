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

## Module 7 detail (done 2026-07-08)

Like Modules 6/6.5, the prompt infrastructure itself needed no honest-skip labs.

Built:
- `docs/modules/07_prompt_engineering_for_small_local_models.md` — theory chapter: why small
  models need stricter prompts, system message discipline, few-shot/negative examples,
  prompt compression, output constraints, injection resistance, versioning, and regression
  tests, tying prompt structure to Module 6.5's prompt-prefix-reuse layout rule and
  cache-invalidation requirement.
- `packages/local_ai_core/prompts/template.py` — `PromptTemplate`, rendering the canonical
  Role/Task/Input contract/Output contract/Rules/Examples/User input structure.
- `few_shot.py` — `FewShotExample`/`NegativeExample` and formatting.
- `registry.py` — `PromptRegistry`, versions immutable once registered.
- `injection_guard.py` — `wrap_untrusted_input()` and a heuristic
  `scan_for_injection_patterns()`, with an explicit test proving (not just claiming) its
  real limit: a rephrased injection attempt is correctly NOT caught.
- `scripts/module_07/prompt_variants.py` — 5 discipline-level variants of the same
  extraction task, monotonically increasing in structure.
- `prompt_runner.py` (Labs 2-3, reuses Module 3's `json_validity` scorer) and `prompt_eval.py`
  (Labs 5-6: regression suite + compression comparison).
- `evals/prompt_regression/extraction_cases.jsonl` — 6 frozen regression cases.
- `notebooks/07_prompt_engineering.ipynb` — **executed end-to-end**; the discipline-level
  comparison is genuinely discriminating (100% invalid JSON for vague/undisciplined
  variants, 0% once rules/examples are present) against a fake model built to exhibit the
  real effect Module 1 §11 documents.
- `reports/module_07_prompt_comparison.md` — deliverable, including an explicit honesty note
  about Lab 6 (below).
- 81 new tests (562 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

**Honesty note, not a bug:** Lab 6's compression-vs-quality comparison ran successfully in
the notebook (76% character reduction, both prompts scoring 100% pass rate) but this does
NOT demonstrate compression is quality-free — it demonstrates the harness correctly measures
character reduction and pass/fail, using a fake runtime that always returns valid output
regardless of prompt content. A fake built to always succeed cannot honestly show a quality
tradeoff it has no capacity to have. Documented explicitly in the report rather than implied
as a real finding.

Deliberately not done in Module 7:
- No real 3-model comparison and no real compression-quality tradeoff — both need an actual
  model whose behavior is sensitive to prompt content; machine constraint. Harness fully
  built and unit-tested; completing this is running two commands on the resourced Mac.

## Module 8 detail (done 2026-07-08)

The full production extraction pipeline — every reliability layer from constrained decoding
through human review — built and fully verified against `FakeRuntime`, like Modules 6/6.5/7.

Built:
- `docs/modules/08_structured_output_and_extraction.md` — theory chapter: why free-form
  output is fragile, JSON mode vs. schema-constrained output, the 4-layer reliability ladder
  (constrained decoding → schema validation → repair retry → human review), the 7-layer
  validation strategy, streaming-vs-structured ("stream prose, buffer structure"), and every
  Gotcha from the curriculum mapped to a specific piece of code that handles it.
- `packages/local_ai_core/extraction/`: `schemas.py` (curriculum's own `InvoiceExtraction`
  verbatim + `PersonExtraction`), `chunking.py` (paragraph/word-boundary-safe chunking +
  conflict-flagging merge), `confidence.py` (deterministic, model-independent scoring —
  explicitly proven to ignore a model's own self-reported confidence), `review_queue.py`,
  `json_parsing.py` (small intentional duplication of Module 3's logic to preserve the
  packages/scripts layering boundary), and `pipeline.py` (the full `ExtractionPipeline`:
  constrained-decoding-first with recorded `FeatureNotSupported` fallback, bounded repair
  retry, chunked extraction, review-queue integration).
- `scripts/module_08/constrained_decoding_runner.py` (Lab 8: text vs. json_schema vs.
  grammar comparison) and `extraction_eval.py` (Lab 7: golden-label evaluation), both reusing
  Module 3's golden extraction set directly rather than duplicating test data.
- `notebooks/08_structured_output_and_extraction.ipynb` — **executed end-to-end**; every
  reliability layer demonstrated live (clean extraction, repair retry, review-queue firing,
  chunked merge, 3-mode comparison with realistic fake capability differences).
- `reports/module_08_structured_output_reliability_report.md` — deliverable, including
  full write-ups of three things this module's own process caught (below).
- 116 new tests (660 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

**Three real issues caught by this module's own tests/process, not by inspection:**
1. `chunking.py`'s hard-split fallback sliced paragraphs by raw character position, which
   split a word in half across a chunk boundary — caught by a test asserting no word is
   lost, fixed by splitting on whitespace instead.
2. `ExtractionPipeline.__init__` used `review_queue or ReviewQueue()` to default an unset
   queue; since `ReviewQueue` defines `__len__`, an empty (falsy) caller-provided queue was
   silently replaced by a different instance, hiding enqueued items from the caller. Fixed
   with an explicit `is not None` check; grepped the repo for the same pattern elsewhere
   (none found).
3. A notebook demo cell claimed to show the review queue firing but didn't, because one risk
   factor alone only downgrades confidence to "medium" (which doesn't trigger review by
   default) — caught while executing the notebook, fixed by adding a second risk factor so
   the demo actually matches its own narrative.

Deliberately not done in Module 8:
- No real 3-model / 3-mode comparison against actual models — machine constraint. Harness
  fully built and unit-tested; completing this is running two commands on the resourced Mac.
- `placeholder_gbnf_grammar()` is explicitly NOT a real, schema-complete GBNF grammar — real
  JSON-Schema-to-GBNF generation is a nontrivial ecosystem tool, out of scope to reimplement;
  the placeholder only exercises the pipeline's grammar-request code path end to end.
- Lab 7/8 are scoped to the 2 of Module 3's 6 golden records that match `PersonExtraction`'s
  schema — the other 4 use different schemas this module doesn't model. Documented as a thin
  sample rather than padded with schema-mismatched data.

## Module 8.5 detail (done 2026-07-08)

Almost no honest-skip surface this module — real SQLite persistence and real budget/
truncation math need no live model at all, unlike most modules.

Built:
- `docs/modules/08_5_conversation_and_context_management.md` — theory chapter: turn
  structure and chat templates, token-aware history accounting, the 5-strategy comparison
  table (drop-oldest/last-N/summarization/importance-weighted/RAG-backed), sticky context,
  SQLite persistence and restart resumption, the conversation-vs-RAG-vs-tool-state
  separation rule, tool-call/tool-result atomicity, and memory deletion.
- `packages/local_ai_core/conversation/`: `turn.py` (shared `Turn` model), `token_budget.py`
  (`ConversationBudget.history_budget`, injected-token-counter design), `truncation.py`
  (`group_turns()` for tool-pair atomicity, `drop_oldest`, `keep_system_plus_last_n`, both
  sticky-aware), `summarizer.py` (`summarize_then_truncate`, injected `summarize_fn`, always
  keeps 1-2 raw turns, falls back to `drop_oldest` if the summary itself still doesn't fit),
  and `session_store.py` (SQLite via stdlib `sqlite3`, no server/dependency, schema with no
  columns for retrieved-doc content or tool state by design).
- `scripts/module_08_5/`: `chat_loop.py` (Lab 1/6: session-persisted chat + `/forget`),
  `force_past_context_window.py` (Lab 3, fully real — no model needed to prove our own
  budget math), `compare_truncation_strategies.py` (Lab 4, fully real).
- `notebooks/08_5_conversation_and_context_management.ipynb` — **executed end-to-end**, with
  almost every cell showing real computed results rather than honest-skips: real SQLite
  restart-persistence, real budget-exceeded-then-resolved numbers, and a real
  early-fact-retention comparison (drop_oldest: lost; summarize_then_truncate: retained).
- `reports/module_08_5_conversation_memory_report.md` — deliverable, including a note about
  adjusting the tool-pairing demo's budget so it shows the pair actually surviving intact
  (more illustrative) rather than both being dropped together (still correct, less useful
  to look at).
- 94 new tests (750 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Deliberately not done in Module 8.5:
- Lab 5 (recall measurement against early turns) — inherently needs a real model's real
  recall behavior to mean anything; machine constraint, pending the resourced Mac.
- Importance-weighted retention and RAG-backed memory are documented as strategies (theory
  doc §5) but not implemented — importance scoring is task-specific and RAG-backed memory is
  Module 11's entire subject, not a few functions bolted onto a conversation module.
- No real chat-template rendering — `chat_loop.py`'s `render_history()` is an explicitly
  labeled simple stand-in; a real adapter (Module 6) owns actual template rendering.

## Module 9 detail (done 2026-07-09)

Almost no honest-skip surface this module — `FakeEmbedder` is a genuine bag-of-words hashing
embedder, so normalization, cosine similarity, Matryoshka truncation, brute-force search,
metadata filtering, and the full evaluation suite are all proven with real (non-fake) numbers.

Built:
- `docs/modules/09_embeddings_from_first_principles.md` — theory chapter covering embedding
  fundamentals, embedding-serving reality (sentence-transformers vs. Ollama), Matryoshka
  truncation, from-scratch implementation, and evaluation, plus an explicit repo-structure
  note reconciling curriculum.md §19's literal path with §8's canonical structure.
- `packages/local_ai_rag/embeddings/`: `embedder.py` (`Embedder` protocol, `normalize()`,
  `cosine_similarity()`, `truncate_embedding()`, `NumpyEmbeddingIndex` brute-force search with
  metadata filtering), `fake.py` (`FakeEmbedder` — SHA-256 feature-hashing bag-of-words
  embedder, a real if crude technique), `ollama_embedder.py` (`OllamaEmbedder`, reuses Module
  6's `LLMError` taxonomy via `map_httpx_error`), `sentence_transformers_embedder.py`
  (`SentenceTransformersEmbedder`, lazy-import + injected `load_fn`/`encode_fn`, same DI
  pattern as Module 6's `MLXRuntime`), `eval.py` (`recall_at_k`, `precision_at_k`,
  `reciprocal_rank`, `ndcg_at_k`, `evaluate_embedder()`, `measure_embedding_throughput()`).
- `scripts/module_09/`: `generate_and_search.py` (Labs 1-4, 6: build a 5-document corpus,
  index it, brute-force search, evaluate recall@k, metadata-filter), `compare_embedding_models.py`
  (Lab 5: compares two `FakeEmbedder` dimensionalities as an honest stand-in for two real
  models, since this machine can't run two real distinct embedders).
- `notebooks/09_embeddings_from_first_principles.ipynb` — **executed end-to-end**, every cell
  showing real computed results: word-overlap similarity, corpus search, metadata filtering,
  full eval metrics, a real dimensionality-vs-ranking-quality comparison, and real throughput
  timing.
- `reports/module_09_embedding_model_report.md` — deliverable, including the real 64d-vs-4d
  comparison showing MRR/nDCG degrade under hash collisions while recall@k stays unaffected.
- 91 new tests (841 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Deliberately not done in Module 9:
- No real neural embedding model run — `OllamaEmbedder` and `SentenceTransformersEmbedder` are
  built and fully unit-tested (against `httpx.MockTransport` and injected fake `load_fn`/
  `encode_fn` respectively) but need a running Ollama server or a downloaded
  sentence-transformers model; machine constraint, pending the resourced 32GB Mac.
- `compare_embedding_models.py` compares two `FakeEmbedder` configurations, not two real
  models — documented explicitly as an honest stand-in rather than silently passed off as a
  real model comparison.

## Module 10 detail (done 2026-07-09)

No honest-skip surface at all this module — Chroma and LanceDB are vector database libraries,
not LLM runtimes or model weights, so both are installed (`uv add chromadb lancedb`,
`onnxruntime<1.20` pinned for macOS 13 wheel compatibility) and every lab runs for real.

Built:
- `docs/modules/10_vector_search_and_local_vector_databases.md` — theory chapter: brute-force
  vs. ANN search, indexing, metadata filters, hybrid search, persistence, incremental
  updates, deletes/reindexing, local vector DB trade-offs, and metadata-first retrieval
  architecture.
- `packages/local_ai_rag/stores/`: `vector_store.py` (`VectorStore` protocol shared by all
  backends), `numpy_store.py` (async wrapper around Module 9's `NumpyEmbeddingIndex`, which
  gained a `delete()` method this module), `chroma_store.py` (real Chroma collection, cosine
  space, `upsert`-based overwrite, `where`-clause metadata filtering, real persistence),
  `lancedb_store.py` (real LanceDB table, cosine metric, `merge_insert`-based overwrite,
  client-side JSON metadata filtering, real persistence), `hybrid.py` (term-overlap keyword
  scoring + Reciprocal Rank Fusion of vector and keyword rankings).
- `scripts/module_10/`: `store_comparison.py` (Labs 1-4, 6: identical corpus across all three
  backends, agreement check, metadata filters, hybrid search recovery), `benchmark_and_evaluate.py`
  (Labs 5-6: real latency measurement and real recall/precision/MRR/nDCG across all three
  backends, reusing Module 9's `eval.py` metric functions).
- `notebooks/10_vector_search_and_local_vector_databases.ipynb` — **executed end-to-end**,
  every cell a real measurement: 3-backend agreement, metadata filtering, hybrid search
  recovery, upsert/delete correctness, real on-disk persistence across a fresh client, and
  real latency/recall numbers.
- `reports/module_10_vector_store_comparison_report.md` — deliverable, including a real
  implementation bug found and fixed while building this module (see below).
- 53 new tests (900 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Real bug found and fixed while building this module: Chroma's plain `collection.add()` and
LanceDB's plain `table.add()` do **not** overwrite an existing document on a duplicate id —
Chroma silently keeps the original document, LanceDB silently appends a duplicate row. Caught
by testing the overwrite-on-same-id contract Module 9's `NumpyEmbeddingIndex` already
established, not assumed to hold. Fixed with each backend's real upsert primitive
(`collection.upsert()`, `table.merge_insert("id").when_matched_update_all()...`).

Deliberately not done in Module 10:
- SQLite + vector extension and DuckDB + Parquet + vectors are documented in the theory doc's
  options table but not implemented — three real backends already prove the `VectorStore`
  protocol is backend-agnostic.
- No ANN-vs-brute-force accuracy divergence demonstrated — the corpus used throughout this
  module (5 documents) is too small for Chroma/LanceDB's ANN indexes to diverge from exact
  search; a larger-scale benchmark is the natural next step.
- Reranking and context packing are Module 12's subject, not implemented here.

## Module 11 detail (done 2026-07-09)

Every stage through prompt assembly runs for real this module against a genuine 20-file
markdown corpus (`datasets/rag_docs/nimbus_handbook/`) — only answer generation needs a live
LLM runtime, wired via Module 6's `LLMRuntime` protocol and fully exercised with `FakeRuntime`.

Built:
- `datasets/rag_docs/nimbus_handbook/` — 20 markdown files, a fictional cloud-storage
  handbook (account management, billing, file sharing, sync clients, API docs, security) with
  internally consistent, uniquely-stated facts, used as real retrieval ground truth.
- `docs/modules/11_rag_v1_naive_rag.md` — theory chapter covering the naive RAG architecture,
  document loading, chunking, retrieval, prompt assembly, basic citations, and the
  curriculum's own gotchas made measurable rather than just documented.
- `packages/local_ai_rag/`: `loaders/markdown_loader.py` (doc-id-from-filename, title/body
  split), `chunkers/document_chunker.py` (wraps Module 8's `chunk_text()`, adds stable
  `chunk_id = f"{doc_id}::{index}"` citation keys), `retrievers/naive_retriever.py`
  (embed-then-search, no reranking/hybrid by design), `context_packers/citation_packer.py`
  (curriculum's minimal RAG prompt verbatim, citation-marker extraction), `pipeline.py`
  (`NaiveRagPipeline` — full ingest/retrieve/answer flow, `RagAnswer.citations_are_grounded`
  detects invented citations by cross-checking against actually-retrieved chunk ids).
- `scripts/module_11/`: `build_and_query.py` (Labs 1-2), `qa_eval.py` (Labs 3-4: 8 answerable
  + 4 unanswerable hand-labeled golden questions, doc-level recall/precision/MRR/nDCG reusing
  Module 9's `eval.py`), `compare_chunk_sizes.py` (Lab 5: same golden set across 3 chunk
  sizes).
- `notebooks/11_rag_v1_naive_rag.ipynb` — **executed end-to-end**, every cell a real
  measurement, including a deliberately provoked invented-citation detection demo.
- `reports/module_11_naive_rag_report.md` — deliverable, including the real (imperfect,
  honest) 0.62 recall@3 number and the real chunk-size-vs-quality comparison.
- 49 new tests (952 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Real proof, not assumed: chunking too aggressively (150 chars) measurably hurts retrieval —
recall@3 drops from 0.62 (at 500 chars) to 0.38, MRR from 0.62 to 0.29. Unanswerable questions
score noticeably lower (0.39-0.51 top score) than answerable ones typically do, a real signal
the retriever isn't confidently wrong on out-of-corpus questions.

Deliberately not done in Module 11:
- No real model's generated answer, and no real-model observation of "the model may ignore
  context" or "the model may answer from prior knowledge" — both need a live LLM to be
  meaningful; pending the resourced 32GB Mac.
- No context-budget enforcement — naive RAG packs all top-k chunks into the prompt regardless
  of size by definition; Module 8.5's budget machinery isn't re-applied here.
- No reranking, hybrid search, or query rewriting in `NaiveRetriever` — Module 10's
  `hybrid_search()` exists and is proven, but naive RAG deliberately doesn't call it; that
  upgrade is explicitly Module 12's subject.
- Text cleaning is minimal (title/body split only) — real-world document parsing (PDFs,
  HTML, OCR) is Module 12's "deeper document parsing."

## Module 12 detail (done 2026-07-09)

Every stage of the production RAG pipeline diagram runs for real this module except final
answer generation (needs a live LLM runtime) and the cross-encoder reranker (needs downloaded
model weights) — both wired via dependency injection and fully unit-tested with fakes.
Document parsing (PDF layout, OCR, parser comparison) is deliberately theory-only — see the
theory doc's "Scope note" — since it needs real messy PDFs and heavy optional dependencies for
a single lab's payoff; `structural_chunker.py` demonstrates the same underlying principle
(structure-aware chunk boundaries) on markdown instead.

Built:
- `docs/modules/12_rag_v2_production_retrieval.md` — theory chapter covering all 16 core
  topics, the production pipeline diagram, the RAG memory note, the context packing strategy,
  and an explicit scope note on document parsing.
- `packages/local_ai_rag/chunkers/`: `parent_child_chunker.py` (small child chunks reference
  large parent chunks), `semantic_chunker.py` (embedding-similarity-based chunk boundaries,
  genuinely different from fixed-size chunking), `structural_chunker.py` (markdown
  tables/fenced code blocks preserved as atomic units — caught and fixed a placeholder-index
  collision bug between the two block types during development).
- `packages/local_ai_rag/retrievers/`: `parent_child_retriever.py` (searches children, returns
  deduplicated parent text), `query_expansion.py` (`rewrite_query`, `multi_query_retrieve` via
  RRF, `hyde_retrieve`), `time_aware.py` (exponential recency decay), `acl.py`
  (`AclAwareRetriever`, predicate-based non-exact-match filtering, over-fetch to preserve `k`).
- `packages/local_ai_rag/rerankers/`: `heuristic_reranker.py` (real, non-neural vector+keyword
  reordering), `cross_encoder_reranker.py` (lazy-import/DI pattern, honest-skip).
- `packages/local_ai_rag/context_packers/budget_packer.py` — the curriculum's exact context
  budget shape, source-diversity-capped packing, and lost-in-the-middle reordering (highest
  relevance at both edges, weakest in the middle); `citation_packer.py` extended with
  `summarize_source_citations()` for document-level citations.
- `packages/local_ai_rag/incremental_indexer.py` — SHA-256 content-hash diffing: unchanged
  documents skip re-embedding entirely, changed documents are fully re-indexed, removed
  documents' chunks are deleted.
- `packages/local_ai_rag/production_pipeline.py` (`ProductionRagPipeline`) — wires
  rewrite/ACL-filter/retrieve/rerank/pack/generate/validate-citations/log-trace into one real
  pipeline over any `Embedder`/`VectorStore`/`LLMRuntime`.
- `scripts/module_12/`: `parent_child_demo.py` (Lab 1), `reranking_demo.py` (Labs 2-5, includes
  a real ACL-tagged restricted document scenario), `incremental_indexing_demo.py` (Lab 6) — all
  run against the Module 11 Nimbus handbook corpus.
- `notebooks/12_rag_v2_production_retrieval.ipynb` — **executed end-to-end**, every cell a real
  measurement.
- `reports/module_12_production_retrieval_report.md` — deliverable, including a real
  distinction between two different citation-grounding failures (ACL leak vs. packing drop)
  and an honest report of where `FakeEmbedder` underserves parent-child retrieval and HyDE.
- 104 new tests (1056 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Real bug found and fixed while building this module: `structural_chunker.py`'s table and code
block extraction both used independently-numbered placeholder indices, causing a collision
that silently duplicated one structural block's content over another's when a document
contained both a table and a code block. Fixed by sharing one placeholder-index list across
both extraction passes; caught by testing "both structures in one document" before shipping.

Deliberately not done in Module 12:
- Document parsing as code (PDF layout extraction, OCR, PyMuPDF/docling/markitdown/unstructured
  comparison) — theory-only, see the theory doc's scope note.
- Real cross-encoder reranking, real LLM-generated query rewrites/multi-queries/HyDE passages —
  fully built and unit-tested, pending the resourced 32GB Mac.
- Query classification (the production pipeline diagram's first stage) — no concrete decision
  to route on with a single corpus and single retrieval strategy.
- Sliding windows — already covered by Module 8's `chunk_text(..., overlap_chars=N)`, reused
  unchanged rather than reimplemented.

## Module 13 detail (done 2026-07-09)

Every metric, statistic, and detector runs for real this module — only `LocalJudge`'s own
verdicts need a live LLM (`FakeRuntime`-backed, fully unit-tested).

Built:
- `evals/rag_eval/nimbus_golden_set.jsonl` — 16 real, hand-authored questions over the Module
  11 Nimbus handbook corpus, curriculum's exact schema.
- `docs/modules/13_rag_v3_evaluation_citations_guardrails.md` — theory chapter covering all 13
  core topics, the judge-model problem, the RAG metrics/failure-taxonomy tables, and an
  architecture note on the retrieval-metrics refactor (see below).
- `packages/local_ai_core/evals/`: `golden_set.py` (`GoldenCase` + JSONL loader),
  `retrieval_metrics.py` (recall/precision/MRR/nDCG **moved from Module 9**, plus Ragas-style
  `context_precision`/`context_recall` aliases), `answer_metrics.py` (must_contain/
  must_not_contain, keyword-overlap relevance, refusal detection), `citation_verifier.py`
  (grounding + chunk-level faithfulness scoring), `hallucination_detection.py` (AUROC
  implemented from scratch, no `scikit-learn` dependency), `local_judge.py` (structured
  faithfulness verdicts via any `LLMRuntime`), `judge_calibration.py` (simple agreement,
  Cohen's kappa), `synthetic_questions.py`, `prompt_injection.py` (7-pattern regex screen).
- `scripts/module_13/`: `common.py` (shared `ScriptedGoldenRuntime` — a controlled generator
  stand-in with 2 deliberately corrupted golden cases), `build_golden_set.py` (Labs 1-2),
  `run_rag_evaluation.py` (Labs 3-4), `citation_and_injection_checks.py` (Labs 5-6),
  `judge_calibration_demo.py` (Labs 7-8).
- `notebooks/13_rag_v3_evaluation_citations_guardrails.ipynb` — **executed end-to-end**, every
  cell a real measurement.
- `reports/module_13_rag_evaluation_report.md` — deliverable, including two real bugs found
  and fixed while building this module (see below) and a real, non-rubber-stamped
  judge-calibration/AUROC result.
- 92 new tests (1148 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Architecture refactor: Module 9's `local_ai_rag/embeddings/eval.py` had defined
`recall_at_k`/`precision_at_k`/`reciprocal_rank`/`ndcg_at_k` as RAG-embedding-specific code, but
they have no embedding-specific coupling. Moved to `local_ai_core/evals/retrieval_metrics.py`
(matching curriculum.md §23's own literal path and §8's canonical structure); `eval.py` now
imports and re-exports them, so existing imports keep working and all 31 of Module 9's original
tests for these functions still pass unmodified against the new source of truth.

Real bugs found and fixed while building this module:
1. `citation_faithfulness_score` counted the citation marker itself (e.g. `password_reset` from
   `[password_reset::0]`) as claim text, artificially inflating overlap with any chunk merely
   *about* that topic. Fixed by stripping the marker before tokenizing.
2. `ScriptedGoldenRuntime`'s citations were appended after the answer's trailing period, which
   simple sentence-splitting treats as a new sentence with no claim text attached — silently
   zeroing out every faithfulness score, even for genuinely well-grounded answers. Fixed by
   moving citation markers inside the sentence, before the period, matching every other
   module's citation convention.

Deliberately not done in Module 13:
- No real LLM-generated judge verdicts, synthetic questions, or generation — pending the
  resourced 32GB Mac.
- No dedicated "context utilization" metric — `must_contain_score` is the practical stand-in;
  a real one would need the same heuristic `citation_faithfulness_score` already provides.
- No separate RAG regression-testing framework — `run_rag_evaluation.py` re-run after any
  pipeline change, diffing the metrics table, is the regression test.

## Module 14 detail (done 2026-07-09)

Almost no honest-skip surface this module — schema validation, the tool registry, permissions,
approval gating, tool budgets, and real SQLite-backed audit logging are all deterministic
Python with zero model dependency. Only LLM-proposed tool selection needs a live LLM.

Built:
- `docs/modules/14_tool_calling_and_deterministic_execution.md` — theory chapter covering all
  12 core topics, the tool execution rule, curriculum's dangerous-tools list, and the four
  tools' real safety mechanisms.
- `packages/local_ai_agents/tools/`: `base.py` (`Tool`/`ToolResult`/error taxonomy),
  `registry.py`, `tool_call.py` (LLM tool-selection proposal parsing), `sandbox.py` (shared
  path-containment logic), `calculator.py` (AST-whitelist safe evaluator, not `eval()`),
  `file_search.py` (curriculum's own `SearchFilesArgs` example), `sql_query.py` (two
  independent read-only defense layers), `write_file.py` (the one dangerous tool, approval-gated).
- `packages/local_ai_agents/policies/`: `permissions.py` (role-based allow/deny),
  `approval.py` (`NullApprovalGate` fails closed, `CallbackApprovalGate` for real use,
  `AutoApprovalGate` tests-only), `budgets.py` (total + per-tool call limits),
  `audit_log.py` (real SQLite persistence, same pattern as Module 8.5's `SessionStore`).
- `packages/local_ai_agents/executors/tool_executor.py` — the deterministic enforcement chain:
  registry → permissions → argument validation → approval (if dangerous) → budget → handler →
  audit log, every attempt logged regardless of outcome.
- `scripts/module_14/`: `tool_registry_demo.py` (Labs 1-4, over the real Nimbus handbook corpus
  and a real SQLite fixture database), `approval_and_dangerous_tools_demo.py` (Labs 5-6).
- `notebooks/14_tool_calling_and_deterministic_execution.ipynb` — **executed end-to-end**,
  including real rejected code-injection and path-traversal payloads.
- `reports/module_14_tool_calling_report.md` — deliverable, including a real pathlib
  absolute-path-override gotcha caught by the sandbox check and two independently-verified SQL
  defense layers.
- 124 new tests (1272 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Real bug caught during development (not shipped): a test-collection module-name collision
(`tools/tests/test_base.py` vs. `runtimes/tests/test_base.py`, and `tools/tests/test_registry.py`
vs. `prompts/tests/test_registry.py`) — the same fix Module 11's `test_pipeline.py` needed,
renamed to `test_tool_base.py`/`test_tool_registry.py` before running the full suite.

Deliberately not done in Module 14:
- No real LLM proposing tool calls — pending the resourced 32GB Mac.
- Only one dangerous tool implemented (`write_file`) of curriculum's nine named categories —
  the mechanism (dangerous flag → approval → audit log) is identical regardless of category.
- `ToolExecutor` doesn't distinguish handler-raised domain errors (e.g. `sql_query.py`'s
  `UnsafeQueryError`) as a separately-typed exception — captured in `ToolResult.error_message`
  and the audit log either way, just not separately catchable by a caller.

## Module 15 detail (done 2026-07-09)

Every planning/execution mechanism is deterministic Python and runs for real this module —
safety budgets, loop prevention, the workflow graph engine, checkpointing, and human approval
interrupts. Only two LLM-dependent pieces (ReAct's reasoning, one bounded workflow decision
point) are scripted-runtime-backed.

Built:
- `docs/modules/15_agentic_workflows_without_chaos.md` — theory chapter covering all 12 core
  topics, the preferred mental model, and the agent safety budget's exact YAML shape.
- `packages/local_ai_agents/planners/`: `safety_budget.py` (`AgentSafetyBudget` - real step/
  tool-call/token counters, real wall-clock runtime), `memory.py` (`AgentMemory` - single-run
  step history), `loop_prevention.py` (`LoopGuard` - a real circuit breaker on repeated
  identical tool calls), `react_loop.py` (`ReActLoop` - the "avoid" shape), `workflow_graph.py`
  (`WorkflowGraph` - one engine for both "state machine" and "graph-based" agent topics, since
  a state machine is a graph with unambiguous branches), `checkpoint_store.py`
  (`CheckpointStore` - real SQLite persistence, same pattern as Module 8.5's `SessionStore`).
- `packages/local_ai_agents/executors/workflow_executor.py` (`WorkflowExecutor`) - runs a
  `WorkflowGraph`: safety budget per step, approval gating for dangerous nodes (Module 14's
  `ApprovalGate`, same fail-closed default), bounded retry-then-fail, and checkpoint-after-every-
  step so a run resumes correctly after an actual restart.
- `scripts/module_15/`: `react_loop_demo.py` (Labs 1-2, includes a real adversarial-prompt
  break), `workflow_graph_demo.py` (Labs 3-4, the same task immune by construction, plus a real
  approval interrupt), `checkpoint_demo.py` (Lab 5), `evaluate_task_success.py` (Lab 6, includes
  a deliberately wrong golden case to prove the scorer discriminates).
- `notebooks/15_agentic_workflows_without_chaos.ipynb` — **executed end-to-end**, every cell a
  real measurement.
- `reports/module_15_agentic_workflows_report.md` — deliverable, including a real checkpoint-
  resume bug found and fixed (see below).
- 73 new tests (1345 total in the repo now, 2 correctly-skipped, all passing); `ruff check .`
  clean.

Real bug found and fixed while building this module: `WorkflowExecutor` originally checkpointed
the node that had *just completed* as the resume point, so resuming re-ran that already-finished
node a second time instead of continuing forward. Caught by a resume test expecting a counter of
2 (1 before a simulated restart, 1 after) that got 3 instead. Fixed by checkpointing the *next*
node computed from the graph's edges, not the one that just finished.

Deliberately not done in Module 15:
- No real LLM driving ReAct's reasoning or the workflow graph's one decision point — pending
  the resourced 32GB Mac.
- Cross-run/long-term agent memory — `AgentMemory` is explicitly scoped to a single run;
  persistent memory is RAG-backed (Module 11) or conversation memory (Module 8.5).
- Only one dangerous workflow node demonstrated — the mechanism is identical regardless of
  which tool triggers it, same scoping decision as Module 14.

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

| Module | Theory doc | Notebook | Code + tests | Deliverable report | Status |
|---|---|---|---|---|---|
| 7. Prompt engineering for small local models | [x] | [x] | [x] | [~] | prompt infra fully built + verified; real 3-model comparison and real compression-quality tradeoff pending a resourced Mac |
| 8. Structured output and extraction | [x] | [x] | [x] | [~] | full reliability-ladder pipeline built + verified via FakeRuntime; real 3-model/3-mode comparison pending a resourced Mac |
| 8.5. Conversation and context management | [x] | [x] | [x] | [x] | complete — SQLite persistence, budget/truncation/summarization all fully verified with real (non-fake) proof; only real recall measurement (Lab 5) pending a resourced Mac |
| 9. Embeddings from first principles | [x] | [x] | [x] | [x] | complete — normalize/cosine/truncation/search/eval all fully verified with real (non-fake) proof; only a real neural embedding model run pending a resourced Mac |
| 10. Vector search and local vector databases | [x] | [x] | [x] | [x] | complete — no honest-skip surface, all three backends (NumPy/Chroma/LanceDB) and hybrid search fully verified with real (non-fake) proof |

## Phase 3 — RAG (Modules 11–13)

| Module | Theory doc | Notebook | Code + tests | Deliverable report | Status |
|---|---|---|---|---|---|
| 11. RAG v1: naive RAG | [x] | [x] | [x] | [x] | complete — loading/chunking/embedding/retrieval/prompt-assembly/citations all fully verified with real (non-fake) proof against a genuine 20-file corpus; only real answer generation pending a resourced Mac |
| 12. RAG v2: production retrieval | [x] | [x] | [x] | [x] | complete — chunking strategies, ACL/time-aware retrieval, heuristic reranking, context packing, source citations, and incremental indexing all fully verified with real (non-fake) proof; real generation and cross-encoder reranking pending a resourced Mac |
| 13. RAG v3: evaluation, citations, and guardrails | [x] | [x] | [x] | [x] | complete — golden dataset, retrieval/answer/citation metrics, AUROC, judge calibration, and injection detection all fully verified with real (non-fake) proof; only real judge/generation quality pending a resourced Mac |

## Phase 4 — Agents/tools (Modules 14–17)

| Module | Theory doc | Notebook | Code + tests | Deliverable report | Status |
|---|---|---|---|---|---|
| 14. Tool calling and deterministic tool execution | [x] | [x] | [x] | [x] | complete — schema validation, registry, permissions, approval, budgets, and audit logging all fully verified with real (non-fake) proof; only LLM-proposed tool selection pending a resourced Mac |
| 15. Agentic workflows without chaos | [x] | [x] | [x] | [x] | complete — safety budgets, loop prevention, workflow graph engine, checkpointing, and approval interrupts all fully verified with real (non-fake) proof, including a real reproduced adversarial-prompt failure; only ReAct reasoning and one workflow decision point pending a resourced Mac |
| 16. MCP and local tool ecosystems | [ ] | [ ] | [ ] | [ ] | not started |
| 17. Local coding assistants | [ ] | [ ] | [ ] | [ ] | not started |

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
