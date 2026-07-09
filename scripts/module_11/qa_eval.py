"""Labs 3-4 — test with answerable and unanswerable questions, measure
retrieval quality manually (against a hand-labeled golden set).

Retrieval is evaluated at the *document* level (a retrieved chunk counts
as a hit if its source doc_id is in the question's relevant_doc_ids) since
chunk_id boundaries shift with chunk size (Lab 5) - doc-level relevance is
the stable ground truth a human actually labeled by reading the corpus.
Reuses Module 9's `eval.py` metric functions rather than reimplementing
recall/precision/MRR/nDCG a third time in this repo.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_rag.embeddings.eval import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.pipeline import NaiveRagPipeline  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "rag_docs" / "nimbus_handbook"


@dataclass(frozen=True)
class GoldenCase:
    question: str
    relevant_doc_ids: frozenset[str]  # empty = unanswerable from this corpus


ANSWERABLE_CASES = [
    GoldenCase("How long does a password reset link stay valid?", frozenset({"password_reset"})),
    GoldenCase("What is Nimbus's refund window after purchase?", frozenset({"refund_policy"})),
    GoldenCase("What happens if a scheduled payment fails?", frozenset({"payment_methods"})),
    GoldenCase("How much storage does the Free plan include?", frozenset({"subscription_plans"})),
    GoldenCase("What is the maximum single-file upload size on the Business plan?", frozenset({"file_upload_limits"})),
    GoldenCase("How many backup codes does 2FA setup provide?", frozenset({"two_factor_authentication"})),
    GoldenCase("Which regions can store Nimbus account data?", frozenset({"supported_regions"})),
    GoldenCase("How long after account deletion is data permanently removed?", frozenset({"account_deletion"})),
]

UNANSWERABLE_CASES = [
    GoldenCase("Who is the CEO of Nimbus?", frozenset()),
    GoldenCase("Does Nimbus offer a student discount?", frozenset()),
    GoldenCase("What is Nimbus's carbon offset program?", frozenset()),
    GoldenCase("What is the maximum number of workspace members on the Enterprise plan?", frozenset()),
]


async def build_pipeline(chunk_max_chars: int = 500) -> NaiveRagPipeline:
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    runtime = FakeRuntime(default_response="I don't know based on the provided documents.")
    pipeline = NaiveRagPipeline(embedder, store, runtime, model="fake-model", chunk_max_chars=chunk_max_chars)
    await pipeline.ingest_directory(CORPUS_DIR)
    return pipeline


async def evaluate_answerable(pipeline: NaiveRagPipeline, k: int = 3) -> dict:
    recalls, precisions, rrs, ndcgs = [], [], [], []
    for case in ANSWERABLE_CASES:
        results = await pipeline.retrieve(case.question, k=k)
        retrieved_doc_ids = [r.metadata["doc_id"] for r in results]
        recalls.append(recall_at_k(retrieved_doc_ids, case.relevant_doc_ids, k))
        precisions.append(precision_at_k(retrieved_doc_ids, case.relevant_doc_ids, k))
        rrs.append(reciprocal_rank(retrieved_doc_ids, case.relevant_doc_ids))
        ndcgs.append(ndcg_at_k(retrieved_doc_ids, case.relevant_doc_ids, k))
    n = len(ANSWERABLE_CASES)
    return {
        "mean_recall_at_k": sum(recalls) / n,
        "mean_precision_at_k": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        "mean_ndcg_at_k": sum(ndcgs) / n,
    }


async def evaluate_unanswerable(pipeline: NaiveRagPipeline, k: int = 3) -> list[dict]:
    """Unanswerable questions have no golden relevant_doc_ids, so there's
    nothing to compute recall against - instead this records the top
    retrieval score for each, a real (if informal) signal of how
    confidently irrelevant the best-available chunk is.
    """
    rows = []
    for case in UNANSWERABLE_CASES:
        results = await pipeline.retrieve(case.question, k=k)
        top_score = results[0].score if results else None
        rows.append({"question": case.question, "top_score": top_score})
    return rows


async def run_lab(k: int = 3) -> dict:
    pipeline = await build_pipeline()
    answerable_metrics = await evaluate_answerable(pipeline, k=k)
    unanswerable_rows = await evaluate_unanswerable(pipeline, k=k)
    return {"answerable_metrics": answerable_metrics, "unanswerable_rows": unanswerable_rows}


def result_to_markdown(result: dict) -> str:
    lines = ["# Labs 3-4 — answerable/unanswerable questions, retrieval quality\n"]
    lines.append("## Answerable questions: retrieval quality (doc-level)\n")
    m = result["answerable_metrics"]
    lines.append(f"- mean recall@k: {m['mean_recall_at_k']:.2f}")
    lines.append(f"- mean precision@k: {m['mean_precision_at_k']:.2f}")
    lines.append(f"- MRR: {m['mrr']:.2f}")
    lines.append(f"- mean nDCG@k: {m['mean_ndcg_at_k']:.2f}")
    lines.append("\n## Unanswerable questions: top retrieval score (no golden match exists)\n")
    for row in result["unanswerable_rows"]:
        score = f"{row['top_score']:.3f}" if row["top_score"] is not None else "n/a"
        lines.append(f"- \"{row['question']}\" -> top score {score}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
