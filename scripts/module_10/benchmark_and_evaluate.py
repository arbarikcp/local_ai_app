"""Labs 5-6 — benchmark retrieval latency and evaluate recall@k across all
three `VectorStore` backends, real timings and real metrics, no honest-skip
(Chroma and LanceDB are libraries installed on this machine, not LLM
runtimes).
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from store_comparison import CORPUS, make_stores, populate  # noqa: E402

from local_ai_rag.embeddings.eval import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.stores.vector_store import VectorStore  # noqa: E402

EVAL_CASES = [
    ("I forgot my password", {"doc_password", "doc_password2"}),
    ("track my shipment", {"doc_shipping"}),
    ("ACC88213", {"doc_order_code"}),
]


async def benchmark_latency(store: VectorStore, query_vectors: list, k: int = 3, repeats: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        for vector in query_vectors:
            await store.search(vector, k=k)
    elapsed = time.perf_counter() - start
    total_queries = repeats * len(query_vectors)
    return elapsed / total_queries if total_queries else 0.0


async def evaluate_recall(store: VectorStore, embedder: FakeEmbedder, k: int = 3) -> dict:
    recalls, precisions, rrs, ndcgs = [], [], [], []
    for query, relevant_ids in EVAL_CASES:
        query_vector = await embedder.embed_query(query)
        results = await store.search(query_vector, k=k)
        retrieved_ids = [r.doc_id for r in results]
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))
    n = len(EVAL_CASES)
    return {
        "mean_recall_at_k": sum(recalls) / n,
        "mean_precision_at_k": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        "mean_ndcg_at_k": sum(ndcgs) / n,
    }


async def run_lab(tmp_dir: str, k: int = 3) -> dict:
    embedder = FakeEmbedder(dimensions=32)
    stores = make_stores(tmp_dir, embedder.dimensions)
    await populate(stores, embedder)

    query_vectors = [await embedder.embed_query(query) for query, _relevant in EVAL_CASES]

    results = {}
    for name, store in stores.items():
        latency = await benchmark_latency(store, query_vectors, k=k)
        metrics = await evaluate_recall(store, embedder, k=k)
        results[name] = {"mean_latency_seconds": latency, **metrics}
    return results


def results_to_markdown_table(results: dict) -> str:
    header = (
        "# Labs 5-6 — retrieval latency and recall@k across backends\n\n"
        f"(corpus size: {len(CORPUS)}, eval cases: {len(EVAL_CASES)})\n\n"
        "| Store | Mean latency (s) | Recall@k | Precision@k | MRR | nDCG@k |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = "\n".join(
        f"| {name} | {r['mean_latency_seconds']:.6f} | {r['mean_recall_at_k']:.2f} | "
        f"{r['mean_precision_at_k']:.2f} | {r['mrr']:.2f} | {r['mean_ndcg_at_k']:.2f} |"
        for name, r in results.items()
    )
    return header + rows + "\n"


def main(argv: list[str] | None = None) -> int:
    tmp_dir = tempfile.mkdtemp(prefix="module10-bench-")
    try:
        results = asyncio.run(run_lab(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(results_to_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
