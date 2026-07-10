"""ModelBoundRuntime (ARCHITECTURE.md "Runtime<->model binding") — binds a
`LLMRuntime` instance to one fixed `model_id`, so the already-real,
unmodified `FallbackRuntime` (Module 20) can carry two different models
(primary and fallback) as its two chain entries without `FallbackRuntime`
itself needing to know about task routing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest, LLMResponse


@dataclass
class ModelBoundRuntime:
    runtime: LLMRuntime
    model_id: str

    def _bind(self, request: LLMRequest) -> LLMRequest:
        return request.model_copy(update={"model": self.model_id})

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self.runtime.generate(self._bind(request))

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async for chunk in self.runtime.stream(self._bind(request)):
            yield chunk

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        return await self.runtime.tokenize(self.model_id, rendered_prompt)
