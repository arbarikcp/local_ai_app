# Module 3 — Local Model Selection and Benchmarking

> Phase: Foundation · Bible reference: [curriculum.md §13](../../curriculum.md#13-module-3--local-model-selection-and-benchmarking)

## Goal

Teach model selection as a repeatable engineering process — not a one-time opinion. The
output of this module is a benchmark harness you keep using for the rest of the course, and
a `models/MODEL_CATALOG.md` that never lets a model's assumed capabilities go unverified.

> **Machine note:** this repo is built on a Mac that must never have a model runtime
> installed ([[project-local-ai-app-curriculum]] constraint, confirmed in Module 2). Every
> piece of this module's harness — datasets, scorers, runner — is built and unit-tested
> against fakes here. Running it against real models happens on the resourced Mac; see
> `reports/module_03_local_model_selection_report.md` for exactly what that leaves pending.

## 1. Reading model cards

A model card should answer, before you pull anything:

- What is the base architecture and parameter count?
- Is this a **base** or **instruct** model (§3 below)?
- What context length is claimed, and was it actually trained/evaluated at that length, or
  extended via a scaling trick (RoPE scaling, YaRN, etc.) that may degrade quality near the
  claimed ceiling?
- What data was it trained on, to the extent disclosed (code-heavy? multilingual? cutoff
  date?)?
- What license governs it, and does that license actually permit your use case (§2)?
- What quantized variants exist, and who produced them (official vs. community GGUF/MLX
  conversions can differ in quality)?

Treat a model card as a set of claims to verify, not facts to import into your architecture.

## 2. License checks

The bible is explicit about this: **do not treat "open weights" as "open license."**

| Family (verify per exact release) | Typical terms | Practical catch |
|---|---|---|
| Qwen 2.5/3 | Often Apache-2.0 | Check each exact model/variant — differs by size and packaging |
| Llama 3.x | Meta Community License | Acceptable-use policy + scale-based restrictions; not OSI-open |
| Gemma | Gemma Terms of Use | Use restrictions apply; not OSI-open |
| Mistral | Mixed: some Apache-2.0, some research/non-commercial | Verify each release |
| Phi | Often permissive for recent small models | Verify per release and source |
| Embedding/reranker models | Mixed | Same seriousness as generator licenses |

This table (from curriculum.md §6.2.1) is a teaching aid, not legal advice, and it decays —
every entry in `models/MODEL_CATALOG.md` records its own `license_notes` and
`last_verified` date rather than trusting a table like this one indefinitely.

## 3. Base vs instruct models

A **base** model is trained only for next-token prediction on a large corpus; it completes
text, it does not reliably follow instructions or hold a conversation. An **instruct**
(or "chat") model has been further tuned (SFT, RLHF/DPO, etc.) to follow instructions and
converse. Almost everything in this course uses instruct variants — benchmarking a base
model against instruction-following tasks will produce misleadingly bad results that reflect
the wrong comparison, not the model's real capability.

## 4. Chat templates

Every instruct model expects its prompt wrapped in a specific chat template — role markers,
special tokens, system/user/assistant separators. Two consequences that matter for
benchmarking specifically:

- Calling a model with the wrong (or no) chat template silently degrades quality — the model
  is technically running but not "reading" the prompt the way it was trained to.
- Runtimes usually apply the correct template for you (Ollama's Modelfile, llama.cpp's
  `--chat-template` / built-in detection, `transformers`' `apply_chat_template`) — the
  benchmark harness in this module calls the runtime's own chat/completion endpoint rather
  than hand-rendering prompts, specifically to avoid reintroducing this class of bug.

## 5. Context length claims

Revisit Module 1 §6: an advertised context window is an architecture capability claim, not a
guarantee of usable quality at that length, and not a guarantee your Mac can afford the KV
cache for it (Module 4 does the memory math). Benchmarking a model's *long-context question
answering* (task 10 below) is how you find out whether the claimed window is actually usable
for your task, independent of whether it fits in memory at all.

## 6. Quantized model variants

The same base model at different quantizations is, for benchmarking purposes, effectively a
**different model** — it can win or lose different tasks at different quantization levels.
`models/MODEL_CATALOG.md` records `quantization_tested` as a list for exactly this reason:
a scorecard is only valid for the quantization it was measured at.

## 7. Model-specific strengths

Model families have known-different strengths (a code-tuned model on code tasks, a model
with stronger multilingual data on non-English tasks, etc.). The benchmark task suite (§9)
exists precisely so these differences show up as measured task scores instead of vibes.

## 8. Task-specific benchmark design

A single "how good is this model" number is close to meaningless for an application. This
course benchmarks **per task**, across the benchmark dimensions below, and only aggregates
after that.

| Category | Metrics |
|---|---|
| Latency | TTFT, total latency, p50, p95, p99 |
| Throughput | tokens/sec, requests/minute |
| Memory | idle memory, peak memory, context memory growth |
| Quality | task score, human score, pass/fail |
| Reliability | invalid JSON rate, timeout rate, refusal rate |
| RAG | context precision, context recall, faithfulness |
| Agent | correct tool selection, valid arguments, step count |

## 9. Human evaluation vs automated evaluation

Automated scorers (exact match, JSON validity, rubric judges) are fast and repeatable but
each has blind spots:

- **Exact/structural match** (§ `exact_match.py`, `json_validity.py`) is precise but brittle
  — it can't grade open-ended quality.
- **Rubric/LLM-as-judge** (§ `rubric_judge.py`) scores open-ended quality but inherits the
  judge model's own biases and blind spots — a small local judge model is not a reliable
  arbiter of a task it would itself fail (the "judge-model problem," covered again in depth
  in Module 13).
- **Human evaluation** is the ground truth for anything subjective, but doesn't scale to
  every regression run.

Practical rule this course follows: automated scorers gate every run (fast feedback,
regression protection); human spot-checks calibrate whether the automated scorers are still
trustworthy, on a sample, periodically — not on every run.

## 10. Regression datasets

A benchmark dataset that changes every time you run it can't tell you if a model, prompt, or
runtime upgrade made things better or worse. The golden sets under `evals/golden_sets/`
built in this module are **frozen, versioned inputs with expected outputs** — the same rule
Module 8 (structured output) and Module 13 (RAG evaluation) will reuse for their own
regression suites.

## Benchmark task suite

This module's harness implements 6 task-type datasets (curriculum's literal deliverable
list); the curriculum's broader 10-item task-suite list also names SQL generation, prompt
injection resistance, and long-context QA as benchmark categories — those are intentionally
deferred to the modules that own them properly (SQL/tool-argument generation deepens in
Module 14; prompt injection resistance deepens in Module 22; long-context QA deepens in
Module 4's context-scaling labs) rather than duplicated shallowly here. The 6 built now
already exceed the assessment's "at least 5 tasks" bar:

1. `summarization.jsonl`
2. `extraction.jsonl`
3. `classification.jsonl`
4. `code.jsonl` (covers both code explanation and test generation)
5. `rag.jsonl` (grounded-answer-from-context, simplified precursor to Module 13's full RAG eval)
6. `tool_calling.jsonl` (tool selection + valid-argument generation, precursor to Module 14)

## Deliverable

```text
models/MODEL_CATALOG.md               # candidate models across categories, license-tracked
evals/golden_sets/*.jsonl             # the 6 task datasets above
scripts/module_03/
  scorers/
    exact_match.py
    json_validity.py
    rag_metrics.py
    rubric_judge.py
  run_benchmark.py
reports/
  model_scorecard_TEMPLATE.md         # reusable blank scorecard
  module_03_local_model_selection_report.md   # this module's own deliverable report
```

Curriculum note: the bible's illustrative deliverable path is `model-eval-suite/...`; this
build follows the repo-wide convention established in Modules 1–2 of placing pre-abstraction
module code under `scripts/module_NN/` and shared eval data under the root `evals/`
directory, rather than a separate nested project folder.

## Assessment

Compare at least 3 models across at least 5 tasks and justify model selection. **This
machine cannot run models**, so `run_benchmark.py` and every scorer are verified against
injected fake generation functions in unit tests; the actual 3-models-×-5-tasks comparison
run and scorecard are pending the resourced Mac — see the deliverable report for the exact
command.
