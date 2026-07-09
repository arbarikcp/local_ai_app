# local_ai_app — Production AI App Development with Local LLMs on Mac

This is the **driving document**: for every module, it says what the module is for, what to
read, what to run, and what has to be installed first — so you always know the next concrete
action. It does not restate the theory.

- **[curriculum.md](curriculum.md)** — the full course bible: goals, theory, labs, deliverables,
  and assessment criteria for all 25 modules, 5 projects, and the capstone. Treat this README
  as the index into it.
- **[PROGRESS.md](PROGRESS.md)** — live status: what's built, what's tested, what's blocked and
  why. Check here before assuming a module is further along than it is.

## Quickstart

```bash
brew install git make cmake python@3.12 uv jq ripgrep   # once, if not already present
uv sync                                                   # install/refresh the Python env
make test                                                 # run all unit tests
make notebook                                             # uv run jupyter lab
```

Everything in this repo runs through `uv run ...` (or the `Makefile` targets, which wrap
`uv run`). Do not `pip install` into a global interpreter — the project's dependencies are
pinned in `pyproject.toml` / `uv.lock`.

### The one thing to know before you start

**This repo is built and practiced on a Mac that must never have a model runtime or model
weights installed on it** (limited disk/memory). Ollama, llama.cpp/llama-cpp-python, and MLX
are deliberately never installed here — that is a standing constraint, not a temporary gap
closed by Module 2. The plan is: build every module's theory/notebook/code/tests here, then
execute the model-running labs later on a separate, properly resourced Mac using the exact
commands each module documents.

Every module's labs that require an actual model run say so explicitly — in the theory doc,
in the notebook output, and in the deliverable report — rather than presenting fabricated
numbers. See `reports/module_01_local_llm_observations.md` and
`reports/module_02_environment_report.md` for what that looks like in practice. If *you* are
running this repo on a properly resourced Mac, you can run those labs for real; just fold the
output into the module's report in place of the "SKIPPED" sections.

## How every module is packaged

Each module follows the same seven-part pattern (curriculum.md, "Final note"):

```text
1. markdown theory chapter   -> docs/modules/NN_name.md
2. runnable notebook         -> notebooks/NN_name.ipynb
3. Python code               -> packages/*/ or scripts/module_NN/ (see note below)
4. unit tests                -> next to the code, run via `make test`
5. project exercise          -> projects/ (once the module contributes to one)
6. deliverable / eval report -> reports/module_NN_*.md
7. production checklist      -> folded into later modules (21-23) once applicable
```

**Code location note:** the reusable `LLMRuntime` abstraction lives in exactly one place —
`packages/local_ai_core/runtimes/`, built in Module 6. Earlier modules (1–5) are
pre-abstraction: they write throwaway/lab-local scripts under `scripts/module_NN/` to observe
real runtime behavior first, on purpose, so the Module 6 abstraction is designed from
evidence rather than guessed upfront. If a module's code would need to reach into another
module's package, that gets flagged before doing it, not silently done.

## Repo layout

```text
curriculum.md      the full bible (source of truth for scope/theory)
README.md          this file — driving/how-to-run doc
PROGRESS.md         live build status
docs/modules/       one theory chapter per module
notebooks/          one runnable notebook per module
scripts/module_NN/  pre-abstraction lab code (Modules 1-5)
packages/           reusable library code (from Module 6 onward)
  local_ai_core/      runtime abstraction, prompts, schemas, evals, tracing, security
  local_ai_rag/        loaders, chunkers, embeddings, stores, retrievers, rerankers
  local_ai_agents/     tools, policies, planners, executors
  local_ai_gateway/    api, routing, streaming, auth, rate limits
projects/           the 5 hands-on projects + capstone
datasets/           test/eval data per domain
evals/              golden sets and regression suites
models/             MODEL_CATALOG.md and benchmark results
reports/            per-module deliverables
```

## Module-by-module guide

Status legend: ✅ built and tested · 🚧 in progress · ⬜ not started. Kept in sync with
[PROGRESS.md](PROGRESS.md) — if they ever disagree, PROGRESS.md is correct.

### Phase: Foundation

#### ✅ Module 1 — Local LLM systems thinking

Builds the mental model everything else depends on: what a local LLM actually consumes in
memory (weights vs. KV cache), why context length is a memory cost not just a length limit,
how to count tokens correctly for local models (never with `tiktoken`), and the specific ways
small (1B–4B) models fail — so later modules read as consequences of these constraints
instead of a grab-bag of framework tutorials.

- **Read:** [docs/modules/01_local_llm_systems_thinking.md](docs/modules/01_local_llm_systems_thinking.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/01_local_llm_basics.ipynb   # memory-math + tokenizer demos, runs with no installs
  uv run python scripts/module_01/lab_1_1_multi_model_run.py    # needs Ollama + pulled models
  uv run python scripts/module_01/lab_1_2_long_prompt_stress_test.py --model qwen2.5:3b
  uv run python scripts/module_01/lab_1_3_small_model_failure_analysis.py --model qwen2.5:1.5b
  uv run pytest scripts/module_01 -q                        # 19 tests, no runtime needed
  ```
- **Install needed:** nothing for the notebook's memory-math and tokenizer-fallback cells or
  for the unit tests. Labs 1.1–1.3 need Ollama running with models pulled — without it they
  print an explicit skip message, they do not fail silently or fake data.
- **Deliverable:** [reports/module_01_local_llm_observations.md](reports/module_01_local_llm_observations.md)
  (infra complete; empirical lab results pending a resourced Mac — see the constraint above).

#### ✅ Module 2 — Mac local AI development environment

Turns a Mac into a reliable local-AI workstation: dev tool check, model-cache accounting, and
smoke tests for all three runtimes (Ollama, llama-cpp-python server, MLX) so Module 1's labs
(and everything after) have a real runtime to run against — on the *other* machine.

- **Read:** [docs/modules/02_mac_local_ai_development_environment.md](docs/modules/02_mac_local_ai_development_environment.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/02_mac_environment_setup.ipynb   # dev-tool check + cache scan + all 3 smoke tests, no installs
  uv run python scripts/module_02/mac_environment_check.py      # standalone dev-tool check
  uv run python scripts/module_02/model_cache.py                # standalone cache report
  uv run python scripts/module_02/smoke_test_runtimes.py         # full combined report (what produced the deliverable)
  uv run pytest scripts/module_02 -q                             # 23 tests, no runtime needed
  ```
- **Install needed:** nothing to read/run the checks themselves — they're read-only. To
  actually complete Labs 2.2–2.4 (on a resourced Mac): `bash scripts/module_02/setup_mac.sh`
  first (brew tools, Ollama, llama-cpp-python, MLX — reviewed line by line before running).
- **Deliverable:** [reports/module_02_environment_report.md](reports/module_02_environment_report.md)
  — Lab 2.1 (dev tools) fully run here and found a real gap (`ripgrep` binary missing, only
  shadowed by a shell function); Labs 2.2–2.4 pending the resourced Mac.

#### ✅ Module 3 — Local model selection and benchmarking

Teaches model selection as a repeatable engineering process — reading model cards, license
checks, and running a benchmark suite (6 task types: summarization, extraction,
classification, code, RAG, tool calling) across candidate models — rather than picking "the
best local model" once and hard-coding it.

- **Read:** [docs/modules/03_local_model_selection_and_benchmarking.md](docs/modules/03_local_model_selection_and_benchmarking.md),
  [models/MODEL_CATALOG.md](models/MODEL_CATALOG.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/03_model_benchmarking.ipynb   # runs the harness against fake models, no installs needed
  uv run python scripts/module_03/run_benchmark.py --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b   # needs Ollama + pulled models
  uv run pytest scripts/module_03 -q                          # 72 tests, no runtime needed
  ```
- **Install needed:** nothing to build/test the harness itself. A real 3-model comparison
  needs Ollama running with 3+ models pulled (see Module 2).
- **Deliverable:** [reports/module_03_local_model_selection_report.md](reports/module_03_local_model_selection_report.md)
  + [reports/model_scorecard_TEMPLATE.md](reports/model_scorecard_TEMPLATE.md) (reusable
  template) — harness fully built and proven against fake models; real comparison pending a
  resourced Mac.

#### ✅ Module 4 — Quantization, context, and memory math

Derives the exact weights + KV-cache memory formulas (already previewed in Module 1's
notebook) and pairs every prediction with a *measured* number, so "this model fits" becomes a
verifiable claim instead of a guess. Includes real, working memory-sampling tooling (proven
against a dummy process, since no model runtime runs on this machine).

- **Read:** [docs/modules/04_quantization_context_and_memory_math.md](docs/modules/04_quantization_context_and_memory_math.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/04_quantization_context_memory_math.ipynb   # reproduces every theory-doc number + proves the memory sampler, no installs needed
  uv run python scripts/module_04/lab_4_1_quantization_comparison.py --tags <tag1> <tag2>   # needs Ollama + models at multiple quantizations
  uv run python scripts/module_04/lab_4_2_context_scaling.py --model qwen2.5:3b
  uv run python scripts/module_04/lab_4_3_concurrency_simulation.py --model qwen2.5:3b
  uv run python scripts/module_04/lab_4_4_predict_then_measure.py --model-tag <tag> --shape qwen2.5-7b   # the core predict-vs-actual deliverable
  uv run pytest scripts/module_04 -q                          # 57 tests, no runtime needed
  ```
- **Install needed:** nothing for the formulas, model-shape registry, or memory sampler
  (real tooling, proven against a dummy subprocess). The four labs need Ollama running with
  models pulled (Lab 4.1 needs the same model at 2+ quantization tags).
- **Deliverable:** [reports/module_04_quantization_context_memory_report.md](reports/module_04_quantization_context_memory_report.md)
  — every formula verified against the theory doc's worked examples (and one real discrepancy
  found: the doc's 128K-context row is a rounded approximation); real measurement pending a
  resourced Mac.

#### ✅ Module 5 — Serving local models

Compares serving patterns (direct CLI, local HTTP API, OpenAI-compatible server, gateway) and
runtime-specific behavior (streaming, warmup, cancellation, error handling) across Ollama,
llama.cpp, and MLX. Includes a documented (not yet measured) 4-runtime × 6-feature comparison
matrix.

- **Read:** [docs/modules/05_serving_local_models.md](docs/modules/05_serving_local_models.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/05_serving_local_models.ipynb   # proves parsers against fixtures + renders the feature matrix, no installs needed
  bash scripts/module_05/serve_ollama.sh qwen2.5:3b             # needs Ollama (Module 2)
  bash scripts/module_05/serve_llamacpp.sh python /path/to/model.gguf   # needs llama-cpp-python[server]
  uv run python scripts/module_05/ollama_metadata.py --model qwen2.5:3b
  uv run python scripts/module_05/llamacpp_openai_streaming.py
  uv run python scripts/module_05/run_mlx_generate.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
  uv run pytest scripts/module_05 -q                            # 59 tests, no runtime needed
  ```
- **Install needed:** nothing for the notebook's fixture-based parser proofs or the feature
  matrix. Live labs need the corresponding runtime from Module 2 (MLX labs also need Apple
  Silicon — confirmed present on this dev machine, not yet confirmed on the target 32GB Mac).
- **Deliverable:** [reports/module_05_runtime_serving_matrix.md](reports/module_05_runtime_serving_matrix.md)
  — feature matrix fully documented; also records a real f-string/ternary bug this module
  caught and fixed via its own test suite. Real per-runtime measurement pending the resourced
  32GB Mac.

#### ✅ Module 6 — Python client architecture

Defines the **one** reusable `LLMRuntime` abstraction (request/response types, streaming
interface, error taxonomy, retries) that every later module builds on — informed by what
Modules 1–5 actually observed runtimes doing, not designed speculatively. Fully built and
verified with no honest-skip labs: `FakeRuntime` + `httpx.MockTransport` exercise every
adapter's real code without a live model runtime.

- **Read:** [docs/modules/06_python_client_architecture.md](docs/modules/06_python_client_architecture.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/06_python_client_architecture.ipynb   # live-demonstrates all 4 adapters, no installs needed
  uv run pytest packages/local_ai_core/runtimes -q                    # 165 passing + 2 correctly-skipped, no runtime needed
  ```
- **Install needed:** nothing — this module's adapters are tested via `FakeRuntime` and
  `httpx.MockTransport`, not a live server. Pointing `OllamaRuntime()`/
  `OpenAICompatibleRuntime()` at a *real* server (to confirm the mocks' assumptions hold) is
  the one thing still pending the resourced 32GB Mac.
- **Deliverable:** [reports/module_06_python_client_architecture_report.md](reports/module_06_python_client_architecture_report.md)
  — includes full write-ups of two real bugs this module's own tests caught (an
  SDK-vs-adapter double-retry composition risk, and a false uniform-capability assumption in
  the contract test itself).

### Phase: Serving/performance foundation

#### ✅ Module 6.5 — Serving concurrency, batching, and caching

Explains why a single unified-memory Mac is often a single-sequence machine, how to measure
queueing/latency under concurrency, and how response/semantic/KV-prefix/embedding caching
avoid recomputation — landing on `max_concurrent_requests: 1` as a deliberate, measured
default rather than an accident. Like Module 6, most of this is fully verified without a live
runtime: `FakeRuntime`'s simulated latency proves queueing and caching for real.

- **Read:** [docs/modules/06_5_serving_concurrency_batching_caching.md](docs/modules/06_5_serving_concurrency_batching_caching.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/06_5_serving_concurrency_batching_caching.ipynb   # live-demonstrates everything, no installs needed
  uv run python scripts/module_06_5/lab_caching_before_after.py                  # runs for real right now, no runtime needed
  uv run python scripts/module_06_5/lab_measure_concurrency.py --model qwen2.5:3b   # needs Ollama
  uv run pytest packages/local_ai_core/gateway scripts/module_06_5 -q             # 88 tests, no runtime needed
  ```
- **Install needed:** nothing for the queue/cache/admission-control infrastructure or the
  caching before/after lab. Real 1/2/4-concurrency measurement needs Ollama running (Module 2).
- **Deliverable:** [reports/module_06_5_serving_concurrency_report.md](reports/module_06_5_serving_concurrency_report.md)
  — includes a real 4.05x caching speedup measurement and a full write-up of a real
  concurrency-accounting bug this module's own tests caught. Real per-runtime concurrency
  measurement pending the resourced 32GB Mac.

#### ⬜ Module 20 — Inference optimization under 8–24 GB RAM

(Built later in sequence, grouped here per the curriculum map.) Optimizes latency, memory,
and reliability for local LLM apps once the basics are in place — see curriculum.md §30.

### Phase: Application primitives

#### ✅ Module 7 — Prompt engineering for small local models

Prompt design under weak-reasoning, limited-context, schema-reliability constraints: system
message discipline, few-shot/negative examples, JSON-only prompting, prompt injection
resistance, versioning, and regression tests. Like Modules 6/6.5, fully built and verified
without a live runtime — only the real 3-model comparison and real compression-quality
tradeoff need one.

- **Read:** [docs/modules/07_prompt_engineering_for_small_local_models.md](docs/modules/07_prompt_engineering_for_small_local_models.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/07_prompt_engineering.ipynb          # live-demonstrates everything, no installs needed
  uv run python scripts/module_07/prompt_runner.py --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b   # needs Ollama
  uv run python scripts/module_07/prompt_eval.py --model qwen2.5:1.5b                              # needs Ollama
  uv run pytest packages/local_ai_core/prompts scripts/module_07 -q  # 81 tests, no runtime needed
  ```
- **Install needed:** nothing for the template/registry/injection-guard infrastructure or the
  discipline-level comparison (proven against a fake model built to exhibit the real effect
  being taught). Real model runs need Ollama (Module 2).
- **Deliverable:** [reports/module_07_prompt_comparison.md](reports/module_07_prompt_comparison.md)
  — includes an explicit honesty note that the compression-quality comparison needs a real
  model to mean anything, and doesn't claim otherwise.

#### ✅ Module 8 — Structured output and extraction

Builds reliable local extraction: constrained decoding (grammar/JSON-schema) as the *primary*
reliability layer, Pydantic validation, repair retries, deterministic (never model-self-reported)
confidence scoring, and human review queues, in that priority order. Like Modules 6/6.5/7,
fully built and verified without a live runtime.

- **Read:** [docs/modules/08_structured_output_and_extraction.md](docs/modules/08_structured_output_and_extraction.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/08_structured_output_and_extraction.ipynb   # live-demonstrates the full reliability ladder, no installs needed
  uv run python scripts/module_08/constrained_decoding_runner.py --model qwen2.5:1.5b   # needs Ollama
  uv run python scripts/module_08/extraction_eval.py --model qwen2.5:1.5b                 # needs Ollama
  uv run pytest packages/local_ai_core/extraction scripts/module_08 -q     # 116 tests, no runtime needed
  ```
- **Install needed:** nothing for the pipeline itself (constrained decoding, repair retry,
  confidence scoring, review queue, chunking are all proven against `FakeRuntime`). Real
  model runs need Ollama (Module 2).
- **Deliverable:** [reports/module_08_structured_output_reliability_report.md](reports/module_08_structured_output_reliability_report.md)
  — includes full write-ups of two real bugs and one caught demo gap from this module's own
  build process.

#### ✅ Module 8.5 — Conversation and context management

Manages multi-turn state within a small context window: token-aware history budgets,
drop-oldest vs. summarization-buffer strategies, SQLite session persistence, and keeping
conversation memory, RAG memory, and tool state separate. Almost no honest-skip surface —
real SQLite persistence and real budget/truncation math need no live model.

- **Read:** [docs/modules/08_5_conversation_and_context_management.md](docs/modules/08_5_conversation_and_context_management.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/08_5_conversation_and_context_management.ipynb   # real SQLite persistence, budget math, and strategy comparison, no installs needed
  uv run python scripts/module_08_5/force_past_context_window.py                # runs for real, no installs needed
  uv run python scripts/module_08_5/compare_truncation_strategies.py            # runs for real, no installs needed
  uv run python scripts/module_08_5/chat_loop.py --model qwen2.5:1.5b           # interactive, needs Ollama
  uv run pytest packages/local_ai_core/conversation scripts/module_08_5 -q      # 94 tests, no runtime needed
  ```
- **Install needed:** nothing for persistence, budgeting, or truncation/summarization
  strategies — all proven with real (non-fake) results. Only the interactive chat loop and
  Lab 5's recall measurement need Ollama.
- **Deliverable:** [reports/module_08_5_conversation_memory_report.md](reports/module_08_5_conversation_memory_report.md)
  — includes a real early-fact-retention comparison (drop_oldest loses it, summarization
  retains it) and real SQLite restart-persistence proof.

#### ✅ Module 9 — Embeddings from first principles

Embeddings from scratch (normalize → cosine similarity → top-k) before any framework,
including embedding-serving reality (sentence-transformers vs. Ollama endpoints) and
Matryoshka-style dimension truncation. Almost no honest-skip surface — `FakeEmbedder` is a
genuine bag-of-words hashing embedder, so retrieval, evaluation, and comparison numbers are
all real, not fabricated. — curriculum.md §19

- **Read:** [docs/modules/09_embeddings_from_first_principles.md](docs/modules/09_embeddings_from_first_principles.md)
- **Run:**
  ```bash
  uv run jupyter lab notebooks/09_embeddings_from_first_principles.ipynb   # real retrieval, evaluation, and comparison, no installs needed
  uv run python scripts/module_09/generate_and_search.py                   # runs for real, no installs needed
  uv run python scripts/module_09/compare_embedding_models.py              # runs for real, no installs needed
  uv run pytest packages/local_ai_rag/embeddings scripts/module_09 -q      # 91 tests, no runtime needed
  ```
- **Install needed:** nothing — normalization, cosine similarity, Matryoshka truncation,
  brute-force search, metadata filtering, and the full recall@k/precision@k/MRR/nDCG@k/latency/
  throughput eval suite are all proven with real (non-fake) results. `OllamaEmbedder` and
  `SentenceTransformersEmbedder` are built and unit-tested but need Ollama or a downloaded
  sentence-transformers model to run for real.
- **Deliverable:** [reports/module_09_embedding_model_report.md](reports/module_09_embedding_model_report.md)
  — includes a real dimensionality-vs-ranking-quality comparison (64d vs. 4d hash collisions
  degrade MRR/nDCG while recall@k stays unaffected) and real throughput timing.

#### ⬜ Module 10 — Vector search and local vector databases

Brute-force vs. ANN search, metadata/ACL-first retrieval design, and trade-offs across
NumPy/Chroma/LanceDB/DuckDB local vector stores. — curriculum.md §20

### Phase: RAG

#### ⬜ Module 11 — RAG v1: naive RAG

Chunk → embed → store → retrieve → prompt → answer, built from scratch, plus the standard
naive-RAG failure modes (bad chunking, irrelevant top-k, ignored context, invented
citations). — curriculum.md §21

#### ⬜ Module 12 — RAG v2: production retrieval

Evolves naive RAG into production-grade retrieval (deeper document parsing, hybrid search,
reranking, context packing). — curriculum.md §22

#### ⬜ Module 13 — RAG v3: evaluation, citations, and guardrails

Treats RAG as a production subsystem: context precision/recall/faithfulness metrics, the
judge-model problem, and citation/guardrail enforcement. — curriculum.md §23

### Phase: Agents/tools

#### ⬜ Module 14 — Tool calling and deterministic tool execution

Safe tool calling where the LLM *proposes* and deterministic code *enforces* — argument
validation, execution boundaries, and the tool error taxonomy. — curriculum.md §24

#### ⬜ Module 15 — Agentic workflows without chaos

Agentic systems as controlled, inspectable workflows (not open-ended autonomy), with
human-approval gates for risky steps. — curriculum.md §25

#### ⬜ Module 16 — MCP and local tool ecosystems

MCP-style tool integration, and the MCP-vs-A2A distinction, without letting protocol
enthusiasm substitute for architecture. — curriculum.md §26

#### ⬜ Module 17 — Local coding assistants

A local, repo-aware coding assistant: repo search, code explanation, test generation, patch
proposals with human approval before writes. — curriculum.md §27

### Phase: Advanced

#### ⬜ Module 18 — Multimodal local applications

Local apps over images, screenshots, scanned PDFs, diagrams, and tables — OCR/layout parsing
plus optional local VLM analysis. — curriculum.md §28

#### ⬜ Module 19 — Fine-tuning, LoRA, and adapters on Mac

When and how to customize a local model (LoRA/adapters) instead of reaching for a bigger
model. — curriculum.md §29

### Phase: Production

#### ⬜ Module 21 — Observability and tracing

Makes local AI apps debuggable: traces that map to user-visible failures, not just raw logs.
— curriculum.md §31

#### ⬜ Module 22 — Security, privacy, and red teaming

Secures local AI apps against realistic threats (prompt injection, unsafe tool execution,
insecure storage) — the module that makes precise the Module 1 point that local ≠ secure. —
curriculum.md §32

#### ⬜ Module 23 — Packaging and deployment

Packages local AI apps for realistic use: reproducible environments, model distribution,
runbooks. — curriculum.md §33

### Projects & capstone

Each project consumes several modules' worth of packages. Status tracked in PROGRESS.md; all
⬜ not started.

| Project | Objective | Curriculum ref |
|---|---|---|
| 1. Local structured extraction service | Production-like service extracting structured fields from documents with a local LLM | §34 |
| 2. Production local RAG service | Local RAG backend over private technical documents, with reranking and citation verification | §35 |
| 3. Local engineering assistant | Repo-aware coding assistant: explain, search, test-generate, patch-propose with human approval | §36 |
| 4. Multimodal document analyst | Analyze scanned documents/screenshots/diagrams with OCR + optional local VLM | §37 |
| 5. Local inference gateway | Production-style gateway: routing, streaming, tracing, fallback across runtimes | §38 |
| Capstone — Local enterprise AI assistant platform | Integrated offline-first platform combining chat, RAG, extraction, tools, evaluation, observability, and security | §39 |

## Working conventions (for whoever picks this up next)

- Every python file gets pytest unit tests before a module is marked done — run `make test`.
- A module is not "done" in PROGRESS.md until its theory doc, notebook (executed, not just
  written), code+tests, and deliverable report all exist — partial credit is recorded
  explicitly (see Module 1's 🚧-flagged deliverable) rather than rounded up.
- No fabricated benchmark numbers, ever. If a runtime/model isn't available to measure
  against, the report says so and gives the exact command to complete it later.
- Module code stays inside its own module/package boundary; cross-module changes get flagged
  before being made, not made silently.
- **Never install a model runtime or model weights on this machine** (see "The one thing to
  know before you start" above) — that work is deferred to a separate, resourced Mac.
- Every completed module gets an annotated git tag (`module-NN`) so any module's finished
  state can be checked out directly: `git checkout module-02`.
