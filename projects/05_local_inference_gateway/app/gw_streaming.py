"""stream_with_fallback (ARCHITECTURE.md "Streaming with fallback") — a
real gap PROPOSAL.md's survey found: `FallbackRuntime` (Module 20) has no
`stream()` method, and no FastAPI streaming endpoint exists anywhere in
the repo to model from. Fallback only applies before the first chunk is
yielded - once tokens have already reached a client, silently restarting
mid-stream on a different model is a different operation, not a
transparent fallback, so a failure after streaming has started propagates
as a stream error instead of being retried.
"""

from __future__ import annotations

from typing import AsyncIterator

from local_ai_core.optimization.fallback import DEFAULT_FALLBACK_ERRORS
from local_ai_core.runtimes.errors import LLMError
from local_ai_core.runtimes.types import LLMRequest

from gw_model_binding import ModelBoundRuntime


async def stream_with_fallback(
    request: LLMRequest,
    *,
    primary: ModelBoundRuntime,
    fallback: ModelBoundRuntime,
    fallback_errors: tuple[type[LLMError], ...] = DEFAULT_FALLBACK_ERRORS,
) -> AsyncIterator[str]:
    primary_iter = primary.stream(request).__aiter__()
    try:
        first_chunk = await primary_iter.__anext__()
    except StopAsyncIteration:
        return
    except fallback_errors:
        async for chunk in fallback.stream(request):
            yield chunk
        return

    yield first_chunk
    async for chunk in primary_iter:
        yield chunk
