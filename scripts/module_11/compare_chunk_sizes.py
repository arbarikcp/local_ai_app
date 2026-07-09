"""Lab 5 — compare 3 chunk sizes' effect on retrieval quality, using the
same golden set `qa_eval.py` defines. Doc-level recall/precision/MRR/nDCG
are comparable across chunk sizes precisely because they're doc-level, not
chunk-level - chunk_id boundaries shift with chunk size, doc_id doesn't.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from qa_eval import build_pipeline, evaluate_answerable  # noqa: E402

CHUNK_SIZES = [150, 500, 1200]


async def compare_chunk_sizes(chunk_sizes: list[int] = CHUNK_SIZES, k: int = 3) -> dict:
    results = {}
    for size in chunk_sizes:
        pipeline = await build_pipeline(chunk_max_chars=size)
        metrics = await evaluate_answerable(pipeline, k=k)
        results[size] = {**metrics, "chunks_in_index": await pipeline.chunk_count()}
    return results


def results_to_markdown_table(results: dict) -> str:
    header = (
        "# Lab 5 — chunk size vs. retrieval quality\n\n"
        "| Chunk size (chars) | Chunks in index | Recall@k | Precision@k | MRR | nDCG@k |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "\n".join(
        f"| {size} | {r['chunks_in_index']} | {r['mean_recall_at_k']:.2f} | "
        f"{r['mean_precision_at_k']:.2f} | {r['mrr']:.2f} | {r['mean_ndcg_at_k']:.2f} |"
        for size, r in results.items()
    )
    return header + rows + "\n"


def main(argv: list[str] | None = None) -> int:
    results = asyncio.run(compare_chunk_sizes())
    print(results_to_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
