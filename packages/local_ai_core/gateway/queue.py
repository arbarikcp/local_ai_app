"""BoundedRequestQueue — admits up to N concurrent requests, queues up to M
waiting, rejects beyond that (theory doc §3: queueing vs. rejection are the
only two honest responses to too much demand; an unbounded queue is
rejection with extra, slower steps).

Reports queue wait time separately from execution time, since that split is
exactly what a concurrency measurement (Lab 3) needs and what Module 4's
raw concurrency simulation didn't have.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


class QueueFullError(Exception):
    """Raised when a request is rejected because max_queue_size is already reached."""

    def __init__(self, waiting: int, max_queue_size: int) -> None:
        super().__init__(f"Queue is full ({waiting}/{max_queue_size} already waiting)")
        self.waiting = waiting
        self.max_queue_size = max_queue_size


@dataclass(frozen=True)
class QueuedResult(Generic[T]):
    result: T
    queue_wait_seconds: float
    execution_seconds: float


class BoundedRequestQueue:
    """Bounds concurrent execution (``max_concurrent``) and rejects new
    admissions once ``max_queue_size`` requests are already waiting.

    A request that can start immediately (a concurrency slot is free) is
    NEVER rejected, regardless of ``max_queue_size`` - ``max_queue_size``
    bounds how many requests may be *waiting*, not how many may run. This
    matters at ``max_queue_size=0``: that means "no waiting room," not "no
    admission at all" - the very next request should still run immediately
    if a slot is free.
    """

    def __init__(self, max_concurrent: int = 1, max_queue_size: int = 10) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        # _running/_waiting are the source of truth for admission decisions;
        # the semaphore is kept in lockstep with _running purely to provide
        # the actual blocking-until-a-slot-frees-up mechanism.
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = 0
        self._waiting = 0
        self._admission_lock = asyncio.Lock()

    @property
    def waiting_count(self) -> int:
        return self._waiting

    @property
    def running_count(self) -> int:
        return self._running

    async def submit(self, fn: Callable[[], Awaitable[T]]) -> QueuedResult[T]:
        """Run ``fn()`` under the concurrency bound, or raise QueueFullError
        immediately if no slot is free AND the queue is already at capacity.
        """
        async with self._admission_lock:
            if self._running < self.max_concurrent:
                self._running += 1
                must_wait = False
            elif self._waiting < self.max_queue_size:
                self._waiting += 1
                must_wait = True
            else:
                raise QueueFullError(self._waiting, self.max_queue_size)

        wait_start = time.perf_counter()
        await self._semaphore.acquire()
        if must_wait:
            async with self._admission_lock:
                self._waiting -= 1
                self._running += 1
        queue_wait_seconds = time.perf_counter() - wait_start

        try:
            exec_start = time.perf_counter()
            result = await fn()
            execution_seconds = time.perf_counter() - exec_start
        finally:
            self._semaphore.release()
            async with self._admission_lock:
                self._running -= 1

        return QueuedResult(
            result=result, queue_wait_seconds=queue_wait_seconds, execution_seconds=execution_seconds
        )
