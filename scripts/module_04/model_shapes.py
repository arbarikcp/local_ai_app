"""Registry of known model architecture shapes, for feeding memory_math.py.

These figures come from each model family's published architecture config
(hidden size / num layers / num KV heads / head dim), not from running the
model — they are documented, not measured, exactly like
models/MODEL_CATALOG.md's entries (Module 3). Verify against the model's own
``config.json`` before trusting a number for a real capacity decision;
architecture details occasionally change between point releases within a
"family" name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    model_id: str
    n_params: float  # raw parameter count, not billions
    n_layers: int
    n_kv_heads: int
    head_dim: int
    source_note: str


# Populated from publicly documented architecture configs. Treat as
# planning-grade (see module docstring) and re-verify per exact release.
KNOWN_SHAPES: dict[str, ModelShape] = {
    "llama3.1-8b": ModelShape(
        model_id="llama3.1-8b",
        n_params=8_000_000_000,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        source_note="Llama 3.1 8B published config (GQA, 8 KV heads) — matches theory doc's worked example",
    ),
    "qwen2.5-7b": ModelShape(
        model_id="qwen2.5-7b",
        n_params=7_600_000_000,
        n_layers=28,
        n_kv_heads=4,
        head_dim=128,
        source_note="Qwen2.5-7B published config (GQA, 4 KV heads)",
    ),
    "qwen2.5-1.5b": ModelShape(
        model_id="qwen2.5-1.5b",
        n_params=1_500_000_000,
        n_layers=28,
        n_kv_heads=2,
        head_dim=128,
        source_note="Qwen2.5-1.5B published config (GQA, 2 KV heads)",
    ),
    "qwen2.5-coder-7b": ModelShape(
        model_id="qwen2.5-coder-7b",
        n_params=7_600_000_000,
        n_layers=28,
        n_kv_heads=4,
        head_dim=128,
        source_note="Qwen2.5-Coder-7B shares the Qwen2.5-7B base architecture",
    ),
}


def get_shape(model_id: str) -> ModelShape:
    if model_id not in KNOWN_SHAPES:
        raise KeyError(
            f"No known shape for {model_id!r}. Known: {sorted(KNOWN_SHAPES)}. "
            "Add an entry to KNOWN_SHAPES (with a source_note) before using an unlisted model."
        )
    return KNOWN_SHAPES[model_id]
