"""Real, deterministic per-page/per-question metrics (PROPOSAL.md "How
success is measured") not already covered by reused evaluation
infrastructure. `field_exact_match()` mirrors Project 1's
`extraction_metrics.field_exact_match()` exactly (a pure dict comparison,
small enough that reusing it across a project boundary would cost more
clarity than it saves - each project owns its own eval metrics file, same
convention Project 2's `rag_eval_metrics.py` established for its one new
metric, memory).
"""

from __future__ import annotations

import psutil

from doc_prompts import extract_page_citations


def current_process_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def field_exact_match(predicted: dict, reference: dict) -> float:
    if not reference:
        return 1.0
    matches = sum(1 for key, value in reference.items() if predicted.get(key) == value)
    return matches / len(reference)


def text_layer_char_count_matches(actual_text: str, expected_text: str) -> bool:
    """PROPOSAL.md's "OCR quality" metric, honestly scoped: this measures
    real PDF text-layer extraction fidelity against a labeled expected
    string, not real OCR accuracy - this machine has no OCR library
    (confirmed by survey), and text-layer extraction stands in for it.
    """
    return actual_text == expected_text


def route_matches_expected(actual_route: str, expected_route: str) -> bool:
    return actual_route == expected_route


def answer_contains_expected_page_citation(answer_text: str, expected_page_id: str) -> bool:
    return expected_page_id in extract_page_citations(answer_text)


def answer_contains_expected_substring(answer_text: str, must_contain: str) -> bool:
    return must_contain in answer_text
