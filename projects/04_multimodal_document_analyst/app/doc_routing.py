"""decide_route — closes the confirmed gap PROPOSAL.md's survey found:
`memory_cost.py`'s real image-token math existed as pure functions but was
never wired into an actual routing decision anywhere in the repo before
this (ARCHITECTURE.md "Routing"). Composes Module 18's `should_use_vlm()`
(the base text-layer-length signal) with `estimate_image_tokens()` /
`estimate_context_budget_impact()` - real numbers attached to the decision,
not just a route label, only when the route actually is VLM (a text-route
page never needs an image rendered at all).
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.multimodal.memory_cost import estimate_context_budget_impact, estimate_image_tokens
from local_ai_core.multimodal.routing import MultimodalRoute, should_use_vlm
from PIL import Image


@dataclass(frozen=True)
class PageRoutingDecision:
    route: MultimodalRoute
    reason: str
    image_tokens: int | None
    context_budget_fraction: float | None


def decide_route(
    text_layer: str,
    *,
    image: Image.Image | None = None,
    context_window: int,
    min_text_chars: int = 40,
) -> PageRoutingDecision:
    base = should_use_vlm(text_layer, min_text_chars=min_text_chars)
    if base.route != MultimodalRoute.VLM:
        return PageRoutingDecision(
            route=base.route,
            reason=base.reason,
            image_tokens=None,
            context_budget_fraction=None,
        )

    if image is None:
        raise ValueError("route resolved to VLM but no rendered page image was provided")

    image_tokens = estimate_image_tokens(image.width, image.height)
    context_budget_fraction = estimate_context_budget_impact(image_tokens, context_window)
    return PageRoutingDecision(
        route=base.route,
        reason=(
            f"{base.reason}; rendered image costs an estimated {image_tokens} tokens "
            f"({context_budget_fraction:.1%} of a {context_window}-token context window)"
        ),
        image_tokens=image_tokens,
        context_budget_fraction=context_budget_fraction,
    )
