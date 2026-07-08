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

**No local model runtime (Ollama / llama.cpp / MLX) is installed on the machine this repo
was built on.** Module 2 is where you install one. Until then, every module's labs that
require an actual model run will say so explicitly — in the theory doc, in the notebook
output, and in the deliverable report — rather than presenting fabricated numbers. See
`reports/module_01_local_llm_observations.md` for what that looks like in practice. If you
*do* have Ollama/llama.cpp/MLX already installed, you can run those labs for real starting
now; just note it in the module's report.

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
  for the unit tests. Labs 1.1–1.3 need Ollama running with models pulled (Module 2 sets this
  up) — without it they print an explicit skip message, they do not fail silently or fake data.
- **Deliverable:** [reports/module_01_local_llm_observations.md](reports/module_01_local_llm_observations.md)
  (infra complete; empirical lab results pending Module 2's install).

#### ⬜ Module 2 — Mac local AI development environment

Turns the Mac into a reliable local-AI workstation: installs and smoke-tests Ollama,
llama-cpp-python, and MLX so Module 1's labs (and everything after) have a real runtime to
run against.

- **Read:** curriculum.md §12 (not yet split into its own `docs/modules/` chapter)
- **Run:** not yet built — will live at `scripts/smoke_test_ollama.py`,
  `scripts/smoke_test_llamacpp_server.py`, `scripts/smoke_test_mlx.py`
- **Install needed:** Homebrew, then Ollama, llama-cpp-python (with Metal support), MLX/mlx-lm
- **Deliverable:** `reports/environment_report.md` (not yet built)

#### ⬜ Module 3 — Local model selection and benchmarking

Teaches model selection as a repeatable engineering process — reading model cards, license
checks, and running a benchmark suite (latency, throughput, memory, quality, JSON validity)
across candidate models — rather than picking "the best local model" once and hard-coding it.

- **Read:** curriculum.md §13
- **Run:** not yet built — will live at `model-eval-suite/runners/run_benchmark.py`
- **Install needed:** the runtime(s) from Module 2, plus candidate models pulled
- **Deliverable:** `reports/model_scorecard.md` (not yet built)

#### ⬜ Module 4 — Quantization, context, and memory math

Derives the exact weights + KV-cache memory formulas (already previewed in Module 1's
notebook) and pairs every prediction with a *measured* number, so "this model fits" becomes a
verifiable claim instead of a guess.

- **Read:** curriculum.md §14
- **Run:** not yet built — predict-then-measure lab against Module 2's runtimes
- **Install needed:** a runtime + at least one model at multiple quantizations
- **Deliverable:** `reports/quantization_context_memory_report.md` (not yet built)

#### ⬜ Module 5 — Serving local models

Compares serving patterns (direct CLI, local HTTP API, OpenAI-compatible server, gateway) and
runtime-specific behavior (streaming, warmup, cancellation, error handling) across Ollama,
llama.cpp, and MLX.

- **Read:** curriculum.md §15
- **Run:** not yet built — `scripts/serve_ollama.sh`, `scripts/serve_llamacpp.sh`,
  `scripts/run_mlx_generate.py`
- **Install needed:** runtimes from Module 2
- **Deliverable:** `reports/runtime_serving_matrix.md` (not yet built)

#### ⬜ Module 6 — Python client architecture

Defines the **one** reusable `LLMRuntime` abstraction (request/response types, streaming
interface, error taxonomy, retries) that every later module builds on — informed by what
Modules 1–5 actually observed runtimes doing, not designed speculatively.

- **Read:** curriculum.md §16
- **Run:** not yet built — will live at `packages/local_ai_core/runtimes/`
- **Install needed:** runtimes from Module 2
- **Deliverable:** `packages/local_ai_core/runtimes/` + `tests/`

### Phase: Serving/performance foundation

#### ⬜ Module 6.5 — Serving concurrency, batching, and caching

Explains why a single unified-memory Mac is often a single-sequence machine, how to measure
queueing/latency under concurrency, and how response/semantic/KV-prefix/embedding caching
avoid recomputation — landing on `max_concurrent_requests: 1` as a deliberate, measured
default rather than an accident.

- **Read:** curriculum.md §16.5
- **Install needed:** Module 6's runtime abstraction
- **Deliverable:** `reports/serving_concurrency_report.md` (not yet built)

#### ⬜ Module 20 — Inference optimization under 8–24 GB RAM

(Built later in sequence, grouped here per the curriculum map.) Optimizes latency, memory,
and reliability for local LLM apps once the basics are in place — see curriculum.md §30.

### Phase: Application primitives

#### ⬜ Module 7 — Prompt engineering for small local models

Prompt design under weak-reasoning, limited-context, schema-reliability constraints: system
message discipline, few-shot/negative examples, JSON-only prompting, prompt injection
resistance, versioning, and regression tests. — curriculum.md §17

#### ⬜ Module 8 — Structured output and extraction

Builds reliable local extraction: constrained decoding (grammar/JSON-schema) as the *primary*
reliability layer, Pydantic validation, repair retries, confidence scoring, and human review
queues, in that priority order. — curriculum.md §18

#### ⬜ Module 8.5 — Conversation and context management

Manages multi-turn state within a small context window: token-aware history budgets,
drop-oldest vs. summarization-buffer strategies, session persistence, and keeping
conversation memory, RAG memory, and tool state separate. — curriculum.md §18.5

#### ⬜ Module 9 — Embeddings from first principles

Embeddings from scratch (normalize → cosine similarity → top-k) before any framework,
including embedding-serving reality (sentence-transformers vs. Ollama endpoints) and
Matryoshka-style dimension truncation. — curriculum.md §19

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
