"""Citation correctness and faithfulness (theory doc §5, §8) -
`citations_are_grounded()` generalizes Module 11/12's
`RagAnswer.citations_are_grounded`/`ProductionRagAnswer.citations_are_grounded`
properties into a standalone function so this module's evaluation harness
doesn't reimplement it a third time. `citation_faithfulness_score()` is a
new, deliberately-labeled heuristic: real NLI/entailment would need a real
model this course doesn't assume is available.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+")
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def citations_are_grounded(citations: list[str], retrieved_ids: list[str]) -> bool:
    """A citation is correct only if it points to something that was
    actually retrieved (theory doc §8) - an invented citation (Module 11's
    "citations may be invented" gotcha) is a citation absent from
    `retrieved_ids`, checkable without a model.
    """
    retrieved_set = set(retrieved_ids)
    return all(citation in retrieved_set for citation in citations)


def _sentence_containing(answer_text: str, citation_marker: str) -> str:
    sentences = _SENTENCE_RE.split(answer_text)
    for sentence in sentences:
        if citation_marker in sentence:
            return sentence
    return answer_text


def citation_faithfulness_score(answer_text: str, chunk_text_by_id: dict[str, str]) -> float:
    """For each `[chunk_id]` citation marker found in `answer_text`, compare
    the sentence it appears in against the cited chunk's actual text (word
    overlap). Returns the mean per-citation score, or 1.0 vacuously if the
    answer has no citations at all (nothing to be unfaithful about).

    Explicitly a heuristic, not true entailment: a faithful paraphrase that
    shares few words with its source will under-score, and a word-salad
    sentence that happens to share vocabulary with the wrong chunk will
    over-score. Real faithfulness checking needs a real model or a trained
    NLI classifier - deferred, same honesty standard as every heuristic in
    this course.
    """
    citation_markers = _CITATION_RE.findall(answer_text)
    if not citation_markers:
        return 1.0

    scores = []
    for marker in citation_markers:
        chunk_text = chunk_text_by_id.get(marker)
        if chunk_text is None:
            scores.append(0.0)
            continue
        sentence = _sentence_containing(answer_text, f"[{marker}]")
        # Strip the citation marker itself before tokenizing - otherwise a
        # marker like "[password_reset::0]" contributes "password"/"reset"
        # as claim words, artificially inflating overlap with any chunk
        # that happens to be *about* password reset regardless of what the
        # sentence actually claims.
        claim_text = sentence.replace(f"[{marker}]", "")
        sentence_words = _tokenize(claim_text)
        chunk_words = _tokenize(chunk_text)
        if not sentence_words or not chunk_words:
            scores.append(0.0)
            continue
        overlap = len(sentence_words & chunk_words)
        scores.append(overlap / len(sentence_words))

    return sum(scores) / len(scores)
