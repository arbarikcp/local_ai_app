"""Labs 3-4 - retrieval metrics, answer metrics, and Ragas-style context
precision/recall over the real Nimbus handbook golden set, run through
Module 12's `ProductionRagPipeline`. Runs for real except generation,
which uses `common.ScriptedGoldenRuntime` - a controlled stand-in with two
deliberately corrupted cases so the metrics below have real failures to
catch, not just clean passes. Also serves as this module's RAG regression
test (theory doc §10) and surfaces Module 12's `TraceLog` per case
(§13, RAG observability).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from common import build_pipeline_and_golden_set  # noqa: E402

from local_ai_core.evals.answer_metrics import (  # noqa: E402
    must_contain_score,
    must_not_contain_score,
    refusal_check,
)
from local_ai_core.evals.citation_verifier import citation_faithfulness_score, citations_are_grounded  # noqa: E402
from local_ai_core.evals.retrieval_metrics import context_precision, context_recall, reciprocal_rank  # noqa: E402


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


async def evaluate_case(pipeline, case, k: int = 5) -> dict:
    result = await pipeline.answer(case.question, k=k)
    retrieved_doc_ids = _dedupe_preserve_order([c.doc_id.split("::")[0] for c in result.packed_chunks])
    chunk_text_by_id = {c.doc_id: c.text for c in result.packed_chunks}

    # Grounding is checked at the *document* level, not exact chunk_id, matching
    # the golden set's own doc-level ground truth (golden_set.py's design note):
    # a scripted "[doc_id::0]" citation should count as grounded if any chunk of
    # that document was retrieved, even if the specific chunk that survived
    # reranking/packing happens to be "::1" because the document split into more
    # than one chunk.
    cited_doc_ids = [c.split("::")[0] for c in result.citations]
    row = {
        "question_id": case.question_id,
        "category": case.category,
        "requires_refusal": case.requires_refusal,
        "answer_text": result.answer_text,
        "must_contain_score": must_contain_score(result.answer_text, case.must_contain),
        "must_not_contain_score": must_not_contain_score(result.answer_text, case.must_not_contain),
        "citations_are_grounded": citations_are_grounded(cited_doc_ids, retrieved_doc_ids),
        "citation_faithfulness_score": citation_faithfulness_score(result.answer_text, chunk_text_by_id),
        "trace": result.trace,
    }

    if case.requires_refusal:
        row["refused"] = refusal_check(result.answer_text)
        row["context_precision"] = None
        row["context_recall"] = None
        row["reciprocal_rank"] = None
    else:
        relevant = set(case.expected_source_ids)
        row["refused"] = None
        row["context_precision"] = context_precision(retrieved_doc_ids, relevant, k)
        row["context_recall"] = context_recall(retrieved_doc_ids, relevant, k)
        row["reciprocal_rank"] = reciprocal_rank(retrieved_doc_ids, relevant)

    return row


async def run_lab(k: int = 5) -> list[dict]:
    pipeline, golden_cases = await build_pipeline_and_golden_set()
    return [await evaluate_case(pipeline, case, k=k) for case in golden_cases]


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if not r["requires_refusal"]]
    unanswerable = [r for r in rows if r["requires_refusal"]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n_cases": len(rows),
        "mean_context_precision": mean([r["context_precision"] for r in answerable]),
        "mean_context_recall": mean([r["context_recall"] for r in answerable]),
        "mean_reciprocal_rank": mean([r["reciprocal_rank"] for r in answerable]),
        "mean_must_contain_score": mean([r["must_contain_score"] for r in rows]),
        "mean_must_not_contain_score": mean([r["must_not_contain_score"] for r in rows]),
        "mean_citation_faithfulness_score": mean([r["citation_faithfulness_score"] for r in rows]),
        "citation_grounding_failures": [r["question_id"] for r in rows if not r["citations_are_grounded"]],
        "refusal_failures": [r["question_id"] for r in unanswerable if not r["refused"]],
    }


def result_to_markdown(rows: list[dict]) -> str:
    summary = summarize(rows)
    lines = ["# Labs 3-4 - retrieval metrics, answer metrics, Ragas-style evaluation\n"]
    lines.append(f"- Cases evaluated: {summary['n_cases']}")
    lines.append(f"- Mean context precision: {summary['mean_context_precision']:.2f}")
    lines.append(f"- Mean context recall: {summary['mean_context_recall']:.2f}")
    lines.append(f"- Mean reciprocal rank (MRR): {summary['mean_reciprocal_rank']:.2f}")
    lines.append(f"- Mean must_contain score: {summary['mean_must_contain_score']:.2f}")
    lines.append(f"- Mean must_not_contain score: {summary['mean_must_not_contain_score']:.2f}")
    lines.append(f"- Mean citation faithfulness score: {summary['mean_citation_faithfulness_score']:.2f}")
    lines.append(f"- Citation-grounding failures (invented citations): {summary['citation_grounding_failures']}")
    lines.append(f"- Refusal failures (answered instead of abstaining): {summary['refusal_failures']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    rows = asyncio.run(run_lab())
    print(result_to_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
