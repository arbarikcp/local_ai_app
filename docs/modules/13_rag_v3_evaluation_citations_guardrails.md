# Module 13 — RAG v3: Evaluation, Citations, and Guardrails

> Phase: RAG · Bible reference: [curriculum.md §23](../../curriculum.md#23-module-13--rag-v3-evaluation-citations-and-guardrails)

## Goal

Evaluate RAG as a production subsystem — close the loop Modules 9-12 opened: every retrieval
and generation stage those modules built gets a real, measurable evaluation harness here.

> **Machine note:** same discipline as every module since 9. `LocalJudge` (the "largest local
> model as judge" strategy) needs a live LLM (`FakeRuntime` here, real adapter unchanged later).
> Everything else — golden-set loading, retrieval/answer/citation metrics, AUROC, judge-human
> agreement (simple agreement, Cohen's kappa), and prompt-injection pattern detection — is pure,
> deterministic code that runs for real, no honest-skip.

## Repo structure note

`packages/local_ai_core/evals/` matches curriculum.md §23's own literal deliverable path *and*
§8's canonical structure (`evals/` lives under `local_ai_core`, not `local_ai_rag`) — no
deviation needed this module, unlike Module 9's embeddings/stores. This also fixes a layering
issue: Module 9's `local_ai_rag/embeddings/eval.py` had defined `recall_at_k`/`precision_at_k`/
`reciprocal_rank`/`ndcg_at_k` as RAG-embedding-specific code, but they're generic
reference-based retrieval metrics with no embedding-specific coupling — exactly the kind of
thing `local_ai_core/evals/` should own so extraction, RAG, and (later) agent evaluation can all
use the same functions. **This module moves them** to
`local_ai_core/evals/retrieval_metrics.py` and updates `eval.py` to import them, instead of
maintaining two copies of the same four functions.

## The judge-model problem

The course constraint (local models under 8-24GB RAM) means a 3-4B model judging another 3-4B
model is often a weak signal. `evals/local_judge.py`'s `LocalJudge` implements the "largest
local model as judge" row of the curriculum's own strategy table — but the **required lesson**
is judged separately: `evals/judge_calibration.py` measures judge-human agreement
(`simple_agreement`, `cohens_kappa`) *before* any judge verdict is trusted. An unvalidated judge
is a random number generator with fluent explanations - this module treats that literally: no
judge output is used in this module's own reported numbers without a calibration step sitting
next to it.

| Strategy | Implemented here |
|---|---|
| Reference-based deterministic metrics | `retrieval_metrics.py`, `answer_metrics.py` — used wherever a golden label exists |
| Largest local model as judge | `local_judge.py` — `FakeRuntime`-backed, real model honest-skip |
| Human evaluation on sampled slice | `evals/rag_eval/nimbus_golden_set.jsonl`'s `expected_answer`/`must_contain` fields stand in for a human-labeled reference in this offline repo |
| Hosted judge for eval development only | not used — this course stays fully local/offline by design |

## Core topics

### 1. Golden question sets

`evals/rag_eval/nimbus_golden_set.jsonl` — 16 real, hand-authored questions over the Module 11
Nimbus handbook corpus, in curriculum's exact schema (`question_id`, `question`,
`expected_answer`, `expected_source_ids`, `must_contain`, `must_not_contain`, `difficulty`,
`category`), loaded by `evals/golden_set.py`'s `load_golden_set()`. A mix of answerable
(various difficulties/categories) and unanswerable questions, extending Module 11's
answerable/unanswerable golden set to the curriculum's richer schema.

### 2. Synthetic question generation

`evals/synthetic_questions.py`'s `generate_questions_from_document()` — one LLM call per
document asking for N candidate questions, parsed into a list. Mechanically real,
`FakeRuntime`-backed; a real model's question *quality* is an empirical claim deferred to the
resourced Mac, same as every other generation-dependent claim since Module 9.

### 3. Retrieval evaluation

`retrieval_metrics.py`'s `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k` (moved
from Module 9, see "Repo structure note").

### 4. Answer evaluation

`answer_metrics.py`: `must_contain_score()` / `must_not_contain_score()` check the golden set's
own `must_contain`/`must_not_contain` fields against a generated answer — deterministic,
reference-based, no judge needed. `keyword_overlap_relevance()` is a crude-but-real, explicitly
labeled heuristic for "does the answer address the question" (word-overlap between question and
answer), not a substitute for `LocalJudge`'s more holistic (but unvalidated-until-calibrated)
verdict.

### 5. Faithfulness

`citation_verifier.py`'s `citation_faithfulness_score()` - the sentence surrounding each
citation marker is compared (word overlap) against the cited chunk's actual text. A citation
whose surrounding sentence shares almost no vocabulary with the chunk it points to is very
likely unfaithful — a real, crude, explicitly-labeled-as-heuristic signal (not true NLI/entailment,
which would need a real model), same honesty standard as Module 9's `FakeEmbedder`.

### 6-7. Context precision and context recall

`retrieval_metrics.py`'s `context_precision` and `context_recall` are named aliases of
`precision_at_k`/`recall_at_k` — Ragas' own terminology for the same math (Lab 4: "Add
Ragas-style evaluation"), not new metrics; naming them this way makes the evaluation report
readable in Ragas' vocabulary without reimplementing math Module 9 already got right.

### 8. Citation correctness

`citation_verifier.py`'s `citations_are_grounded()` — generalized out of Module 11/12's
`RagAnswer.citations_are_grounded`/`ProductionRagAnswer.citations_are_grounded` properties into
a standalone, reusable function (a citation is correct only if it points to a chunk that was
actually retrieved) so this module's evaluation harness doesn't reimplement it a third time.

### 9. Hallucination detection

`hallucination_detection.py`'s `compute_auroc()` — implemented from scratch (rank-based
Mann-Whitney U formulation, no `sklearn` dependency) since the underlying math is a few lines
and this course avoids adding a heavy dependency for one function. Lab 8 treats hallucination
detection as binary classification: is `citation_faithfulness_score` (or `citations_are_grounded`
as a 0/1 score) separating known-grounded from known-hallucinated answers, measured by AUROC
rather than assumed.

### 10. RAG regression testing

`scripts/module_13/run_rag_evaluation.py` running the full golden set through
`ProductionRagPipeline` and reporting aggregate metrics *is* the regression test — re-running it
after any retrieval/chunking/prompt change and diffing the metrics table is exactly what
regression testing means here. No separate snapshot-diffing framework is built; the golden set
plus the metric functions already are the regression harness.

### 11. Prompt injection from documents

`prompt_injection.py`'s `detect_prompt_injection_patterns()` — a real, if necessarily
incomplete, regex-based screen for common injection phrasings ("ignore previous instructions",
"disregard the above", "you are now", "reveal the system prompt", etc.) that might appear
*inside a retrieved document* rather than in the user's own message — the exact threat model
curriculum names ("a malicious document changes model behavior"). Explicitly documented as a
pattern screen, not a guarantee — a sufficiently rephrased injection defeats it, same honesty
standard applied to every heuristic in this course.

### 12. Refusal behavior

`answer_metrics.py`'s `refusal_check()` — for a golden case with `expected_answer` indicating
"I don't know," checks whether the generated answer actually refuses (matches a refusal
phrase) rather than confidently answering from the model's prior knowledge (curriculum's own
"the model may answer from prior knowledge" gotcha from Module 11, now checkable against a
labeled unanswerable-question set).

### 13. RAG observability

Module 12's `TraceLog` (rewrite/ACL/retrieve/rerank/pack counts) already *is* this module's
observability primitive — `scripts/module_13/run_rag_evaluation.py` surfaces it per golden
case rather than building a second, separate tracing system.

## RAG evaluation dataset

Curriculum's exact shape, implemented by `evals/golden_set.py`'s `GoldenCase`:

```json
{
  "question_id": "q_001",
  "question": "How do I rotate API keys?",
  "expected_answer": "...",
  "expected_source_ids": ["doc_7#chunk_3"],
  "must_contain": ["rotate", "API key"],
  "must_not_contain": ["contact support"],
  "difficulty": "medium",
  "category": "procedural"
}
```

## RAG metrics implemented

| Metric | Implemented as |
|---|---|
| recall@k | `retrieval_metrics.recall_at_k` |
| precision@k | `retrieval_metrics.precision_at_k` |
| MRR | `retrieval_metrics.reciprocal_rank` |
| context precision | `retrieval_metrics.context_precision` (alias) |
| context recall | `retrieval_metrics.context_recall` (alias) |
| citation accuracy | `citation_verifier.citations_are_grounded` |
| faithfulness | `citation_verifier.citation_faithfulness_score` (heuristic) |
| abstention accuracy | `answer_metrics.refusal_check` |
| answer relevance | `answer_metrics.keyword_overlap_relevance` (heuristic) |
| adherence/completeness | `answer_metrics.must_contain_score` / `must_not_contain_score` |
| hallucination detector AUROC | `hallucination_detection.compute_auroc` |
| judge-human agreement | `judge_calibration.simple_agreement`, `judge_calibration.cohens_kappa` |

`context utilization` (did the answer use the relevant context supplied) is documented but not
separately implemented — `must_contain_score` against the golden answer's key facts is this
module's practical stand-in; a dedicated context-utilization metric would need the same
faithfulness-style heuristic `citation_faithfulness_score` already provides.

## RAG failure taxonomy

Curriculum's table, cross-referenced to where each failure is *measurable* in this repo's code,
not just named:

| Failure | Measurable via |
|---|---|
| no retrieval / weak retrieval | `recall_at_k`, `reciprocal_rank` |
| noisy retrieval | `precision_at_k` |
| wrong chunking | Module 12's chunk-size/strategy comparisons |
| context overflow | Module 12's `budget_packer.py` |
| hallucinated answer | `citation_faithfulness_score`, `LocalJudge` |
| hallucinated citation | `citations_are_grounded` |
| stale answer | Module 12's `time_aware.py` |
| ACL leak | Module 12's `acl.py` — and Module 12's own report already caught this failure mode via `citations_are_grounded` |
| injection success | `prompt_injection.detect_prompt_injection_patterns` |

## Hands-on labs

1. **Create golden RAG dataset** — `evals/rag_eval/nimbus_golden_set.jsonl`,
   `scripts/module_13/build_golden_set.py`.
2. **Run retrieval metrics** — `scripts/module_13/run_rag_evaluation.py`.
3. **Run answer metrics** — same script.
4. **Add Ragas-style evaluation** — `context_precision`/`context_recall` naming, same script.
5. **Add citation verifier** — `scripts/module_13/citation_and_injection_checks.py`.
6. **Add malicious document tests** — same script, a real injected Nimbus-handbook-style
   document.
7. **Calibrate a local judge against human labels** — `scripts/module_13/judge_calibration_demo.py`.
8. **Report hallucination detection as binary classification with AUROC** — same script.

## Deliverable

```text
evals/rag_eval/nimbus_golden_set.jsonl
packages/local_ai_core/evals/
  golden_set.py
  retrieval_metrics.py
  answer_metrics.py
  citation_verifier.py
  hallucination_detection.py
  local_judge.py
  judge_calibration.py
  synthetic_questions.py
  prompt_injection.py
  tests/
scripts/module_13/
  build_golden_set.py
  run_rag_evaluation.py
  citation_and_injection_checks.py
  judge_calibration_demo.py
reports/module_13_rag_evaluation_report.md
```
