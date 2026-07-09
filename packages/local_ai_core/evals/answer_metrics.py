"""Answer evaluation (theory doc §4, §12) - reference-based checks against
a golden case's `must_contain`/`must_not_contain` fields (deterministic, no
judge needed), plus a crude-but-real keyword-overlap relevance heuristic
and refusal detection for unanswerable questions.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+")

DEFAULT_REFUSAL_PHRASES = ("i don't know", "i do not know")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def must_contain_score(answer_text: str, must_contain: list[str]) -> float:
    """Fraction of required phrases actually present (case-insensitive
    substring match) - 1.0 vacuously when nothing is required.
    """
    if not must_contain:
        return 1.0
    answer_lower = answer_text.lower()
    hits = sum(1 for phrase in must_contain if phrase.lower() in answer_lower)
    return hits / len(must_contain)


def must_not_contain_score(answer_text: str, must_not_contain: list[str]) -> float:
    """1.0 if none of the forbidden phrases appear, else 0.0 - a single
    forbidden phrase (e.g. a wrong number, an invented policy) makes the
    whole answer wrong, not "partially" wrong.
    """
    if not must_not_contain:
        return 1.0
    answer_lower = answer_text.lower()
    return 0.0 if any(phrase.lower() in answer_lower for phrase in must_not_contain) else 1.0


def keyword_overlap_relevance(question: str, answer_text: str) -> float:
    """Crude, explicitly-labeled heuristic for "does the answer address the
    question": fraction of the question's content words that also appear in
    the answer. Not a substitute for `LocalJudge`'s more holistic verdict -
    a real answer can be relevant while reusing almost none of the
    question's own wording (paraphrase), so this under-scores good answers
    as often as it over-scores bad ones. Useful as a cheap, always-available
    reference-free signal, not as ground truth.
    """
    question_words = _tokenize(question)
    answer_words = _tokenize(answer_text)
    if not question_words:
        return 0.0
    hits = len(question_words & answer_words)
    return hits / len(question_words)


def refusal_check(answer_text: str, refusal_phrases: tuple[str, ...] = DEFAULT_REFUSAL_PHRASES) -> bool:
    """True if the answer contains a recognizable refusal phrase - used
    against golden cases where `requires_refusal` is True (theory doc §12)
    to check the system abstains instead of answering from the model's
    prior knowledge (Module 11's "the model may answer from prior
    knowledge" gotcha, now checkable).
    """
    answer_lower = answer_text.lower()
    return any(phrase in answer_lower for phrase in refusal_phrases)
