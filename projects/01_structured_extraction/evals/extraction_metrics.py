"""Real, deterministic extraction metrics (PROPOSAL.md "How success is
measured") — curriculum's own metric set, computed against a labeled
dataset. Every function here is a pure comparison between a predicted
fields dict and a reference fields dict; no model dependency at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalExample:
    id: str
    schema_name: str
    text: str
    reference: dict


def field_exact_match(predicted: dict, reference: dict) -> float:
    """Fraction of reference fields (including correctly-predicted nulls)
    the predicted dict matches exactly. 1.0 vacuously when the reference
    has no fields.
    """
    if not reference:
        return 1.0
    matches = sum(1 for key, value in reference.items() if predicted.get(key) == value)
    return matches / len(reference)


def missing_field_rate(predicted: dict, reference: dict) -> float:
    """Of the reference fields that genuinely have a value (non-null),
    what fraction did the prediction fail to fill in? 0.0 vacuously when
    the reference has no required (non-null) fields.
    """
    required = [key for key, value in reference.items() if value is not None]
    if not required:
        return 0.0
    missing = sum(1 for key in required if predicted.get(key) is None)
    return missing / len(required)


def hallucinated_field_rate(predicted: dict, reference: dict) -> float:
    """Of the reference fields that are genuinely null (not present in the
    source text), what fraction did the prediction fabricate a non-null
    value for? 0.0 vacuously when the reference has no null fields.
    """
    null_fields = [key for key, value in reference.items() if value is None]
    if not null_fields:
        return 0.0
    hallucinated = sum(1 for key in null_fields if predicted.get(key) is not None)
    return hallucinated / len(null_fields)
