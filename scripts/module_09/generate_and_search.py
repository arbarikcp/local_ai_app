"""Labs 1-4, 6 — generate embeddings, store in NumPy, search by brute
force, evaluate recall@k, add metadata filtering.

Runs for real against FakeEmbedder (a genuine, if crude, bag-of-words
hashing embedder - see embeddings/fake.py) by default: no live model
runtime needed to prove this infrastructure works, unlike most modules.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_rag.embeddings.embedder import NumpyEmbeddingIndex  # noqa: E402
from local_ai_rag.embeddings.eval import EmbeddingEvalCase, evaluate_embedder  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402

CORPUS: dict[str, tuple[str, dict[str, str]]] = {
    "doc_password": ("How to reset your password", {"category": "account"}),
    "doc_billing": ("Update your billing information and payment method", {"category": "billing"}),
    "doc_shipping": ("Track your shipment and delivery status", {"category": "shipping"}),
    "doc_password2": ("Forgot password recovery steps for your account", {"category": "account"}),
    "doc_returns": ("How to return a product for a refund", {"category": "billing"}),
}

EVAL_CASES = [
    EmbeddingEvalCase(query="I forgot my password", positive_doc_ids=["doc_password", "doc_password2"]),
    EmbeddingEvalCase(query="how do I get a refund", positive_doc_ids=["doc_returns"]),
]


async def build_index(embedder) -> NumpyEmbeddingIndex:
    index = NumpyEmbeddingIndex()
    texts = [text for text, _metadata in CORPUS.values()]
    vectors = await embedder.embed_documents(texts)
    for (doc_id, (text, metadata)), vector in zip(CORPUS.items(), vectors):
        index.add(doc_id, text, vector, metadata=metadata)
    return index


async def run_lab(embedder, k: int = 3) -> dict:
    index = await build_index(embedder)

    query_text = "I forgot my password"
    query_vector = await embedder.embed_query(query_text)
    search_results = index.search(query_vector, k=k)
    filtered_results = index.search(query_vector, k=k, metadata_filter={"category": "account"})

    eval_summary = await evaluate_embedder(embedder, index, EVAL_CASES, k=k)

    return {
        "index_size": len(index),
        "query": query_text,
        "top_result_doc_id": search_results[0].doc_id if search_results else None,
        "top_result_score": search_results[0].score if search_results else None,
        "filtered_result_doc_ids": [r.doc_id for r in filtered_results],
        "mean_recall_at_k": eval_summary.mean_recall_at_k,
        "mean_precision_at_k": eval_summary.mean_precision_at_k,
        "mrr": eval_summary.mrr,
        "mean_ndcg_at_k": eval_summary.mean_ndcg_at_k,
    }


def result_to_markdown(result: dict) -> str:
    top_score = f"{result['top_result_score']:.3f}" if result["top_result_score"] is not None else "n/a"
    return (
        "# Labs 1-4, 6 — generate, store, search, evaluate, filter\n\n"
        f"- Indexed documents: {result['index_size']}\n"
        f"- Query: `{result['query']}`\n"
        f"- Top result: {result['top_result_doc_id']} (score={top_score})\n"
        f"- Metadata-filtered results (category=account): {result['filtered_result_doc_ids']}\n"
        f"- Mean recall@k: {result['mean_recall_at_k']:.2f}\n"
        f"- Mean precision@k: {result['mean_precision_at_k']:.2f}\n"
        f"- MRR: {result['mrr']:.2f}\n"
        f"- Mean nDCG@k: {result['mean_ndcg_at_k']:.2f}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dimensions", type=int, default=64)
    args = parser.parse_args(argv)

    embedder = FakeEmbedder(dimensions=args.dimensions)
    result = asyncio.run(run_lab(embedder, k=args.k))
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
