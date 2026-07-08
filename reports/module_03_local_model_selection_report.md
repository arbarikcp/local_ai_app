# Module 3 deliverable — local model selection report

Status: **benchmark harness built, unit-tested, and demonstrated against a fake model.
Real multi-model comparison pending a resourced Mac** (standing constraint — see
`PROGRESS.md`).

## What's built and verified

| Artifact | Verified how |
|---|---|
| `models/MODEL_CATALOG.md` | 13 candidate model entries across chat/code/embedding/reranker categories, each with the curriculum's YAML schema (§6.4), license notes, and an explicit "documented, not benchmarked" verification status |
| `scripts/module_03/scorers/exact_match.py` | 8 unit tests |
| `scripts/module_03/scorers/json_validity.py` | 12 unit tests, including the markdown-fence-stripping behavior small models actually exhibit (Module 1 §11) |
| `scripts/module_03/scorers/rag_metrics.py` | 10 unit tests |
| `scripts/module_03/scorers/rubric_judge.py` | 11 unit tests, including score-parse-failure handling |
| `evals/golden_sets/*.jsonl` (6 files, 36 records total) | schema-checked by `test_run_benchmark.py::test_golden_set_files_exist_and_parse`; every record's prompt-building path exercised by `test_prompt_for_record_builds_a_nonempty_prompt_for_every_record` |
| `scripts/module_03/run_benchmark.py` | 22 unit tests covering every scorer dispatch path (including all 5 tool-call-scoring branches), dataset loading, full-benchmark orchestration with injected fake models, and scorecard/comparison table rendering |
| `notebooks/03_model_benchmarking.ipynb` | **Executed end-to-end** — ran the full harness against two fake models and produced genuinely discriminating scores (not a rubber stamp), then correctly skipped the real-model section |
| `reports/model_scorecard_TEMPLATE.md` | Reusable blank scorecard per curriculum's template |

72 new unit tests this module (114 total across the repo), `ruff check .` clean.

## Proof the harness discriminates (from the executed notebook, fake-model run)

| Model | summarization | extraction | classification | code | rag | tool_calling |
|---|---:|---:|---:|---:|---:|---:|
| fake-model-a | 0.00 | 0.33 | 1.00 | 0.00 | 1.00 | 0.20 |
| fake-model-b | 0.00 | 0.17 | 1.00 | 0.00 | 1.00 | 0.20 |

This is a deliberately imperfect fake model (see the notebook's `fake_model` function) —
the point is that scores vary meaningfully by task and by model instead of all reading 1.0
or 0.0, which is what proves the scoring logic actually works rather than trivially passing.
`summarization` and `code` read 0.00 here only because the fake model returns generic text
that doesn't contain the required facts/substrings the `contains_all` scorer checks for —
expected behavior for a fake model, not a harness bug.

## Model catalog summary

13 entries in `models/MODEL_CATALOG.md`, all marked `verification_status: documented` (public
model-card information only — none benchmarked on this machine):

- **Chat/instruction:** Qwen2.5 (1.5B, 7B instruct), Llama 3.1 8B instruct, Gemma 2 9B
  instruct, Phi-3.5 mini instruct
- **Code:** Qwen2.5-Coder (1.5B, 7B)
- **Embedding:** nomic-embed-text, BAAI/bge-small-en-v1.5
- **Reranker:** BAAI/bge-reranker-base

Each entry records its own license caveat rather than trusting a single point-in-time table
— per curriculum.md §6.2.1, license terms must be re-verified per exact release before any
commercial use, and this catalog does not constitute that verification.

## Assessment self-check

> "Student must compare at least 3 models across at least 5 tasks and justify model
> selection."

- **Task coverage: satisfied in the harness** — 6 task types are implemented (exceeds the
  5-task minimum), each with its own scorer and golden set.
- **3-model comparison: not yet done for real.** This machine cannot run models. Once on a
  resourced Mac:
  ```bash
  ollama pull qwen2.5:1.5b
  ollama pull qwen2.5:3b
  ollama pull qwen2.5:7b
  uv run python scripts/module_03/run_benchmark.py --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b \
      > reports/module_03_benchmark_raw_output.md
  ```
  Then fold the resulting comparison table into this report (replacing this section) and
  fill one `reports/model_scorecard_TEMPLATE.md` copy per model, including a written
  justification of which model you'd actually pick for which use case — the harness produces
  numbers, but the "justify model selection" requirement is a judgment call a report has to
  make explicitly, not something the numbers do automatically.

## Known limitations of this module's harness (by design, not oversights)

- No latency/memory/TTFT measurement yet — those benchmark dimensions require a live runtime
  and are captured by `ollama_probe.GenerationObservation` (Module 1) when
  `default_generate_fn` is actually exercised; `run_benchmark.py` doesn't currently thread
  those metrics into the scorecard tables. Worth revisiting once real runs are possible.
- `rubric_judge.py` is built and tested but not wired into `run_benchmark.py`'s scorer
  dispatch yet — none of the 6 golden sets currently use `"scorer": "rubric_judge"`. Adding
  an open-ended task (e.g. "explain this concept") that needs LLM-as-judge scoring is a
  natural extension once a judge model is actually available to call.
- RAG/tool-calling scoring here is intentionally simplified (Module 3 theory doc §"Benchmark
  task suite") — full RAG evaluation is Module 13, full tool-calling evaluation is Module 14.
