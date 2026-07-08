import pytest

from scorers.rag_metrics import (
    answer_is_grounded_refusal,
    citation_validity,
    context_precision,
    context_recall,
    extract_citations,
)


def test_context_precision_fraction_relevant_among_retrieved():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3", "doc9"]
    assert context_precision(retrieved, relevant) == pytest.approx(2 / 3)


def test_context_precision_empty_retrieved_is_zero():
    assert context_precision([], ["doc1"]) == 0.0


def test_context_recall_fraction_relevant_found():
    retrieved = ["doc1", "doc2"]
    relevant = ["doc1", "doc3"]
    assert context_recall(retrieved, relevant) == pytest.approx(0.5)


def test_context_recall_empty_relevant_is_zero():
    assert context_recall(["doc1"], []) == 0.0


def test_extract_citations_finds_bracketed_ids():
    answer = "The tower was completed in 1889 [doc1] and is in Paris [doc2]."
    assert extract_citations(answer) == ["doc1", "doc2"]


def test_extract_citations_empty_when_no_citations():
    assert extract_citations("No citations here.") == []


def test_citation_validity_all_valid():
    answer = "Fact one [doc1], fact two [doc2]."
    assert citation_validity(answer, ["doc1", "doc2", "doc3"]) == 1.0


def test_citation_validity_partial():
    answer = "Fact one [doc1], fabricated fact [doc99]."
    assert citation_validity(answer, ["doc1", "doc2"]) == pytest.approx(0.5)


def test_citation_validity_vacuously_true_with_no_citations():
    assert citation_validity("No citations at all.", ["doc1"]) == 1.0


def test_answer_is_grounded_refusal_detects_refusal_phrase():
    answer = "I don't know based on the provided documents."
    assert answer_is_grounded_refusal(answer, "I don't know based on the provided documents") is True


def test_answer_is_grounded_refusal_false_when_answer_given():
    answer = "The tower was completed in 1889."
    assert answer_is_grounded_refusal(answer, "I don't know based on the provided documents") is False
