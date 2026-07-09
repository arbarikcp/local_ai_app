"""Memory cost of images (theory doc §10) - extends Module 4's memory-math
discipline to images: a real formula, not a rule of thumb. Patch-based
token estimation is the mechanism real ViT-style vision encoders use
(divide the image into fixed-size patches, one token per patch), so this
is a genuine approximation of how VLM image-token costs scale, not an
arbitrary number.
"""

from __future__ import annotations

import math


def estimate_image_tokens(width: int, height: int, patch_size: int = 14) -> int:
    """`ceil(width / patch_size) * ceil(height / patch_size)` - the same
    patch-grid arithmetic a ViT-style vision encoder uses. `patch_size=14`
    matches a common real default (e.g. CLIP/SigLIP-style encoders); a
    caller targeting a specific model should pass that model's real patch
    size instead of trusting the default.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    cols = math.ceil(width / patch_size)
    rows = math.ceil(height / patch_size)
    return cols * rows


def estimate_context_budget_impact(image_tokens: int, context_window: int) -> float:
    """What fraction of a given context window a single image would
    consume - real division, makes "a high-resolution image can cost as
    many tokens as several pages of text" (theory doc §10) a checkable
    number instead of an assertion.
    """
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    return image_tokens / context_window
