"""Real image preprocessing (theory doc §3) via Pillow - grayscale
conversion, contrast enhancement, max-dimension resizing, and rotation by
a *given* angle. Deliberately does not implement automatic skew-*angle
detection* (a real computer-vision problem needing OpenCV or a trained
model, out of scope) - documented as a real gap, not silently approximated.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """`factor > 1.0` increases contrast, `factor < 1.0` decreases it,
    `factor == 1.0` returns the original contrast unchanged - Pillow's own
    `ImageEnhance.Contrast` convention.
    """
    return ImageEnhance.Contrast(image).enhance(factor)


def resize_max_dimension(image: Image.Image, max_dimension: int) -> Image.Image:
    """Scales down (never up) so the longer side is at most
    `max_dimension`, preserving aspect ratio - the real preprocessing step
    that keeps a VLM's real image-token cost bounded (§10, `memory_cost.py`).
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image.copy()
    scale = max_dimension / longest
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def rotate(image: Image.Image, angle_degrees: float) -> Image.Image:
    """Rotates by a *known* angle (e.g. a page orientation metadata field,
    or a user-supplied correction) - real, but not skew *detection*; this
    function does not estimate what angle a crooked scan needs.
    """
    return image.rotate(angle_degrees, expand=True, fillcolor="white")
