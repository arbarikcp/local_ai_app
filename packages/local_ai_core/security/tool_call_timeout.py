"""Per-tool-call timeout enforcement (theory doc §12) — the one gap Module
14's `ToolExecutor` left open: nothing previously bounded how long a tool
handler may run. Reuses `runtimes/errors.py`'s `RequestTimeout` rather than
declaring a fourth timeout type, since a hung tool call and a hung model
call are the same failure mode from a caller's perspective - both mean
"this didn't come back in time," and both should be catchable the same way.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from local_ai_core.runtimes.errors import RequestTimeout

T = TypeVar("T")


async def with_timeout(fn: Callable[[], Awaitable[T]], *, timeout_seconds: float) -> T:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        return await asyncio.wait_for(fn(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RequestTimeout(f"tool call exceeded {timeout_seconds}s timeout", cause=exc) from exc
