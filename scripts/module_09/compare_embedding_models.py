"""Lab 5 — compare two embedding models.

This machine cannot run two real distinct embedding models (no runtime or
weights installed here - see the repo's machine constraint), so the
comparison uses two FakeEmbedder dimensionalities: 64 (few hash
collisions) vs. 4 (severe hash collisions). This is an honest stand-in
that still exercises the real comparison harness and demonstrates a real
effect (dimensionality vs. retrieval quality) rather than faking numbers.
Real usage would pass two real Embedder instances, e.g.
OllamaEmbedder("nomic-embed-text") vs. SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5").
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from generate_and_search import CORPUS, EVAL_CASES, build_index  # noqa: E402

from local_ai_rag.embeddings.embedder import Embedder  # noqa: E402
from local_ai_rag.embeddings.eval import EvalSummary, evaluate_embedder  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402


async def compare_embedders(embedders: dict[str, Embedder], k: int = 3) -> dict[str, EvalSummary]:
    results: dict[str, EvalSummary] = {}
    for name, embedder in embedders.items():
        index = await build_index(embedder)
        results[name] = await evaluate_embedder(embedder, index, EVAL_CASES, k=k)
    return results


def results_to_markdown_table(results: dict[str, EvalSummary]) -> str:
    header = (
        "# Lab 5 — compare two embedding models\n\n"
        f"(corpus size: {len(CORPUS)}, eval cases: {len(EVAL_CASES)})\n\n"
        "| Model | Recall@k | Precision@k | MRR | nDCG@k |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    rows = "\n".join(
        f"| {name} | {s.mean_recall_at_k:.2f} | {s.mean_precision_at_k:.2f} | "
        f"{s.mrr:.2f} | {s.mean_ndcg_at_k:.2f} |"
        for name, s in results.items()
    )
    return header + rows + "\n"


def main(argv: list[str] | None = None) -> int:
    embedders: dict[str, Embedder] = {
        "fake-64d": FakeEmbedder(dimensions=64),
        "fake-4d (severe hash collisions)": FakeEmbedder(dimensions=4),
    }
    results = asyncio.run(compare_embedders(embedders))
    print(results_to_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
