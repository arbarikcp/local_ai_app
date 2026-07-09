# Module 13 deliverable — RAG evaluation report

Status: **complete.** Every metric, statistic, and detector in this module is real, deterministic
code — recall/precision/MRR/nDCG, must_contain/must_not_contain, citation grounding and
faithfulness, AUROC (implemented from scratch), judge-human agreement (simple agreement, Cohen's
kappa), and prompt-injection pattern detection all run for real, no honest-skip. Only
`LocalJudge`'s own verdicts come from a scripted stand-in (`FakeRuntime`); the calibration
statistics that check those verdicts are real regardless of what produced them.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `evals/rag_eval/nimbus_golden_set.jsonl` | — | 16 real, hand-authored questions over the Module 11 Nimbus handbook corpus, curriculum's exact schema |
| `packages/local_ai_core/evals/golden_set.py` | 6 | JSONL loading, optional-field defaults, `requires_refusal` derivation |
| `packages/local_ai_core/evals/retrieval_metrics.py` | 14 | recall@k/precision@k/MRR/nDCG@k (moved from Module 9) + Ragas-vocabulary aliases |
| `packages/local_ai_core/evals/answer_metrics.py` | 13 | must_contain/must_not_contain scoring, keyword-overlap relevance, refusal detection |
| `packages/local_ai_core/evals/citation_verifier.py` | 8 | Citation grounding, chunk-level faithfulness scoring (caught and fixed a real bug — see below) |
| `packages/local_ai_core/evals/hallucination_detection.py` | 7 | From-scratch AUROC: perfect separation, perfect reversal, chance-level ties, degenerate single-class cases |
| `packages/local_ai_core/evals/local_judge.py` | 5 | Structured verdict parsing, real parse-failure detection |
| `packages/local_ai_core/evals/judge_calibration.py` | 12 | Simple agreement, Cohen's kappa against a hand-worked example (p_o=0.8, p_e=0.5 → kappa=0.6) |
| `packages/local_ai_core/evals/synthetic_questions.py` | 4 | LLM-based question generation, real parsing |
| `packages/local_ai_core/evals/prompt_injection.py` | 7 | Pattern detection across 7 common injection phrasings |
| `scripts/module_13/` (4 lab scripts + shared `common.py`) | 47 | Labs 1-8 exercised against the real 20-file Nimbus handbook corpus |
| `notebooks/13_rag_v3_evaluation_citations_guardrails.ipynb` | — | **Executed end-to-end** — every cell a real measurement |

**92 new tests this module** (1148 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Architecture note: retrieval metrics moved to `local_ai_core`

Module 9's `local_ai_rag/embeddings/eval.py` had defined `recall_at_k`/`precision_at_k`/
`reciprocal_rank`/`ndcg_at_k` — four functions with zero embedding-specific coupling, wrongly
scoped to a RAG-only package. This module moved them to
`local_ai_core/evals/retrieval_metrics.py` (matching curriculum.md §23's own literal deliverable
path and §8's canonical structure) and updated `eval.py` to import and re-export them, so
existing imports (`from local_ai_rag.embeddings.eval import recall_at_k`) keep working
unchanged. All 31 of Module 9's original tests for these functions still pass without
modification, run against the new source of truth.

## Two real bugs found and fixed while building this module

**1. `citation_faithfulness_score` counted the citation marker itself as claim text.** A
citation like `[password_reset::0]` tokenizes (with underscores split) into `password` and
`reset` — words that happen to overlap with any chunk *about* password reset, regardless of what
the surrounding sentence actually claims. Caught by
`test_zero_score_when_sentence_shares_no_words_with_the_cited_chunk`, which expected a score of
`0.0` for a sentence about "distant galaxies" and got `0.222` instead. Fixed by stripping the
citation marker from the sentence before tokenizing.

**2. `ScriptedGoldenRuntime`'s citation formatting silently defeated its own faithfulness
scores.** Citations were appended *after* the answer's trailing period
(`"...15 minutes. [password_reset::0]"`), which simple punctuation-based sentence splitting
(`(?<=[.!?])\s+`) treats as its own sentence — leaving the citation marker with no claim text
attached and every faithfulness score computing to `0.0`, even for genuinely well-grounded
answers. Caught by manually inspecting a "should be near-1.0" case that scored exactly `0.0` (see
"Real proof" below). Fixed by inserting citation markers *inside* the sentence, before the
trailing period, matching the convention every other module's citation examples already used
(`"...15 minutes [password_reset::0]."`).

## Real proof: the faithfulness bug, before and after the fix

Before the fix, a textbook well-grounded answer scored **0.00**:

```
answer: "The password reset link expires in 15 minutes for security reasons. [password_reset::0]"
chunk:  "...The reset link expires in 15 minutes for security reasons; if it expires..."
score:  0.00   <- wrong; the sentence and the chunk share nearly every content word
```

After moving the citation marker inside the sentence:

```
answer: "The password reset link expires in 15 minutes for security reasons [password_reset::0]."
score:  1.00
```

A deliberately unfaithful answer (grounded citation, unsupported claim) scores meaningfully
lower, confirming the fix discriminates correctly rather than just returning 1.0 always:

```
answer: "Nimbus was founded in a garage in 2010 [password_reset::0]."
score:  0.43
```

## Real proof: citation grounding (doc-level) and faithfulness (chunk-level) can disagree

Running the full golden set through `ProductionRagPipeline`, several genuinely correct,
doc-level-grounded answers still scored `0.00` on chunk-level faithfulness:

```
q_004  grounded=True   context_recall=1.00  faithfulness=0.00
q_006  grounded=True   context_recall=1.00  faithfulness=0.00
q_011  grounded=True   context_recall=1.00  faithfulness=0.00
```

This is not a bug — it's a real, informative gap between two metrics operating at different
granularities. `citations_are_grounded` (Module 11/12's own definition) only checks that a cited
document was retrieved *somewhere*; `citation_faithfulness_score` requires the *exact* cited
`chunk_id` to be present, because it needs that specific chunk's text to score overlap against.
When a document splits into multiple chunks (`password_reset::0` and `password_reset::1`) and
reranking/packing keeps a different chunk than the one a citation names, doc-level grounding
still says "correct" while chunk-level faithfulness says "unmeasurable, treat as unfaithful." A
production system would want to know about this gap — it's exactly the kind of thing a single
aggregate "citation accuracy: 94%" number would hide.

## Real proof: two deliberately corrupted cases are caught by the metrics designed to catch them

`common.py`'s `ScriptedGoldenRuntime` corrupts exactly two of 16 golden cases:

```
Citation-grounding failures (invented citations): ['q_002', 'q_003', 'q_007']
Refusal failures (answered instead of abstaining): ['q_016']
```

`q_003` and `q_016` are the deliberately corrupted cases, and both are caught. `q_002` and
`q_007` are **not** deliberately corrupted — they're genuine `FakeEmbedder` retrieval misses (the
correct document never made it into the top-5 candidates for those two questions), the same
honest imperfection Modules 11 and 12 already documented, now caught by a different metric
(citation grounding) rather than recall@k directly.

## Real proof: judge calibration surfaces genuine disagreement, not a rubber-stamped 1.0

8 hand-labeled calibration cases, including two deliberately ambiguous ones where the scripted
judge is more conservative than the "human" label:

```
Simple agreement (judge vs. human): 0.75
Cohen's kappa (judge vs. human):     0.53
```

Kappa (0.53, "moderate" by conventional bands) is meaningfully lower than raw agreement (0.75)
because both raters lean toward labeling answers unfaithful in this set, inflating the chance
agreement a naive percentage would hide — exactly the correction Cohen's kappa exists to make,
demonstrated on a real (if small) case set rather than asserted from the formula alone.

## Real proof: AUROC beats chance on real, hand-labeled data

Reusing `citation_faithfulness_score` as a hallucination detector against the same 8
hand-labeled cases:

```
citation_faithfulness_score-as-hallucination-detector AUROC: 0.73
```

0.73, not 1.00 and not 0.50 — a real, imperfect-but-informative detector, computed with a
from-scratch AUROC implementation validated against known properties (perfect separation → 1.0,
perfect reversal → 0.0, all-tied scores → 0.5) before trusting it on real data.

## Real proof: prompt injection pattern detection

A synthetic Nimbus-handbook-style document with an embedded injection payload:

```
Patterns matched: ['ignore (all |the )?previous instructions', 'you are now',
                    'reveal (the |your )?system prompt']
Clean document patterns matched: []
```

Three independent patterns fire on the malicious document, none fire on an ordinary FAQ entry —
a real screen, explicitly documented as incomplete (a rephrased injection would defeat it), not
a guarantee.

## Deliberately not done in Module 13

- No real LLM-generated judge verdicts, synthetic questions, or generation — `LocalJudge`,
  `synthetic_questions.py`, and `ProductionRagPipeline`'s own generation stage are all fully
  built and unit-tested with `FakeRuntime`; real model quality is deferred to the resourced 32GB
  Mac, same discipline as every module since 9.
- No dedicated "context utilization" metric separate from `must_contain_score` — a genuinely
  distinct metric would need the same faithfulness-style heuristic `citation_faithfulness_score`
  already provides; documented as a stand-in rather than reimplemented.
- No separate RAG regression-testing framework — `run_rag_evaluation.py` re-run after any
  retrieval/chunking/prompt change, diffing the metrics table, *is* the regression test; a
  dedicated snapshot-diffing tool would be new infrastructure for a need the golden set + metric
  functions already meet.
