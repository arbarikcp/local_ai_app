"""VisionLanguageModel Protocol + FakeVLM (theory doc §2) - same
dependency-injection pattern as Module 6's `MLXRuntime` / Module 9's
`SentenceTransformersEmbedder`: a real adapter's model-loading call is
lazy-imported inside a function body, tests inject a fake, real model
honest-skip (this repo's machine constraint: no model runtime or weights
installed here at all).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Protocol

from PIL import Image

LoadFn = Callable[[str], Any]
DescribeFn = Callable[[Any, Image.Image, str], str]


class VisionLanguageModel(Protocol):
    async def describe(self, image: Image.Image, prompt: str) -> str: ...


def _real_load(model_id: str) -> Any:
    from mlx_vlm import load

    return load(model_id)


def _real_describe(model: Any, image: Image.Image, prompt: str) -> str:
    from mlx_vlm import generate

    model_obj, processor = model
    return generate(model_obj, processor, prompt=prompt, image=image)


class MlxVisionLanguageModel:
    """Enabling this for real:
        1. In pyproject.toml, uncomment ``"mlx-vlm>=0.1"``, then run ``uv sync``.
        2. Pick a model, e.g. ``"mlx-community/Qwen2-VL-2B-Instruct-4bit"`` -
           no separate download step, ``mlx_vlm.load(model_id)`` downloads
           and caches it on first use.
        3. Construct with no overrides: ``MlxVisionLanguageModel(model_id)`` -
           ``load_fn``/``describe_fn`` already default to the real ``mlx_vlm``
           calls (``_real_load``, ``_real_describe`` above); only tests
           inject fakes.
    """

    def __init__(self, model_id: str, *, load_fn: LoadFn = _real_load, describe_fn: DescribeFn = _real_describe) -> None:
        self.model_id = model_id
        self._load_fn = load_fn
        self._describe_fn = describe_fn
        self._model: Any | None = None

    async def _get_model(self) -> Any:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_fn, self.model_id)
        return self._model

    async def describe(self, image: Image.Image, prompt: str) -> str:
        model = await self._get_model()
        return await asyncio.to_thread(self._describe_fn, model, image, prompt)


class FakeVLM:
    """Deterministic fake for tests - returns a scripted response
    regardless of image content (a VLM's actual visual reasoning can't be
    faked meaningfully, only its call shape), tracks call history for
    assertions.
    """

    def __init__(self, default_response: str = "This is a fake VLM description.") -> None:
        self.default_response = default_response
        self.calls: list[tuple[Image.Image, str]] = []

    async def describe(self, image: Image.Image, prompt: str) -> str:
        self.calls.append((image, prompt))
        return self.default_response
