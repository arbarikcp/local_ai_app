"""The recommended pipeline principle (theory doc's own diagram, §11 "When
not to use a VLM"), made into one real, testable decision function instead
of a rule of thumb:

    Document/image
      -> OCR/layout extraction
      -> deterministic preprocessing
      -> text-based local LLM
      -> VLM only for visual reasoning gaps
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MultimodalRoute(Enum):
    TEXT_LLM = "text_llm"
    VLM = "vlm"


@dataclass(frozen=True)
class RoutingDecision:
    route: MultimodalRoute
    reason: str


def should_use_vlm(text_layer: str, min_text_chars: int = 40) -> RoutingDecision:
    """A document with a usable text layer (real embedded text, or real
    OCR output if OCR were available) never needs a VLM at all - route it
    to a text-based local LLM, which is cheaper and more reliable for the
    same question. Only a document with no extractable text (or one whose
    text layer is too sparse to be useful - `min_text_chars` is a real,
    adjustable threshold, not a magic number) escalates to a VLM.
    """
    stripped = text_layer.strip()
    if len(stripped) >= min_text_chars:
        return RoutingDecision(
            route=MultimodalRoute.TEXT_LLM,
            reason=f"text layer has {len(stripped)} chars (>= {min_text_chars} threshold) - a VLM is unnecessary",
        )
    return RoutingDecision(
        route=MultimodalRoute.VLM,
        reason=f"text layer has only {len(stripped)} chars (< {min_text_chars} threshold) - likely scanned/image-only",
    )
