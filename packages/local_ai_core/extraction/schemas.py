"""Extraction schemas (theory doc §3-4).

Pydantic models double as the JSON Schema source
(``Model.model_json_schema()``) AND the validator, so the schema requested
for constrained decoding and the schema used to validate the result are
guaranteed identical - never two hand-maintained copies that can drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InvoiceExtraction(BaseModel):
    """The curriculum's own example schema (curriculum.md §18), verbatim."""

    invoice_number: str | None = None
    vendor_name: str | None = None
    invoice_date: str | None = Field(default=None, description="ISO date if present")
    currency: str | None = None
    total_amount: float | None = None
    confidence: Literal["low", "medium", "high"]
    evidence: dict[str, str] = Field(default_factory=dict)


class PersonExtraction(BaseModel):
    """The name/age/city task used as a running example since Module 1."""

    name: str | None = None
    age: int | None = None
    city: str | None = None
