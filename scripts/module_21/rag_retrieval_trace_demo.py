"""Lab 4 - trace RAG retrieval. Runs a real retrieval against Module 9's
`NumpyEmbeddingIndex` (no model needed - embeddings are hand-built unit
vectors, same fixture style as Module 9's own tests) and records the real
result into a trace via `record_retrieval_step()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

import numpy as np  # noqa: E402

from local_ai_core.tracing.metrics_registry import MetricsRegistry  # noqa: E402
from local_ai_core.tracing.trace import TraceBuilder  # noqa: E402
from local_ai_rag.embeddings.embedder import NumpyEmbeddingIndex  # noqa: E402

DOCS = [
    ("password-reset-guide", [1.0, 0.0, 0.0], "Password reset links expire after 24 hours."),
    ("billing-faq", [0.0, 1.0, 0.0], "Refunds are issued within 5 business days."),
    ("security-faq", [0.0, 0.0, 1.0], "Enable two-factor authentication in account settings."),
]


def run_lab() -> dict:
    index = NumpyEmbeddingIndex()
    for doc_id, vector, text in DOCS:
        index.add(doc_id=doc_id, text=text, embedding=np.array(vector), metadata={"doc_id": doc_id})

    query_vector = np.array([0.9, 0.1, 0.0])  # closest to password-reset-guide
    results = index.search(query_vector, k=2)

    registry = MetricsRegistry()
    builder = TraceBuilder(request_id="req-rag-001")
    builder.record_retrieval_step(
        query="how do I reset my password?",
        chunk_ids=[r.doc_id for r in results],
        reranker_scores=[r.score for r in results],
    )
    registry.increment("request_count")
    registry.observe("retrieval_recall_estimate", 1.0 if results and results[0].doc_id == "password-reset-guide" else 0.0)

    trace = builder.build()
    return {
        "span_names": trace.span_names(),
        "retrieved_doc_ids": [r.doc_id for r in results],
        "top_score": results[0].score if results else None,
        "metrics_summary": {name: s.count for name, s in registry.summary().items() if hasattr(s, "count")},
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 4 - trace RAG retrieval\n\n"
        f"- Retrieved doc IDs: {result['retrieved_doc_ids']}\n"
        f"- Top score: {result['top_score']:.4f}\n"
        f"- Trace spans recorded: {result['span_names']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
