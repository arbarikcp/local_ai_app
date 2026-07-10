"""The one metric curriculum's evaluation list names that no existing eval
infrastructure computes anywhere in this repo (confirmed by survey):
memory. Real `psutil`-based process RSS measurement - the other seven
metrics (recall@k, precision@k, citation correctness, faithfulness, answer
relevance, abstention accuracy, latency) are all real reuse from
`local_ai_core/evals/` and `rag_query_service.py`, composed in
`run_rag_eval.py`, not reimplemented here.
"""

from __future__ import annotations

import psutil


def current_process_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)
