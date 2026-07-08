import asyncio

import pytest

from local_ai_core.gateway.queue import BoundedRequestQueue, QueueFullError, QueuedResult


class TestConstruction:
    def test_rejects_max_concurrent_below_one(self):
        with pytest.raises(ValueError):
            BoundedRequestQueue(max_concurrent=0)

    def test_rejects_negative_max_queue_size(self):
        with pytest.raises(ValueError):
            BoundedRequestQueue(max_concurrent=1, max_queue_size=-1)

    def test_allows_zero_max_queue_size(self):
        # max_queue_size=0 means "no waiting room" - every request beyond
        # max_concurrent is rejected immediately. Valid, if aggressive.
        BoundedRequestQueue(max_concurrent=1, max_queue_size=0)


class TestSingleRequest:
    async def test_returns_the_function_result(self):
        q = BoundedRequestQueue(max_concurrent=1)

        async def fn():
            return "hello"

        result = await q.submit(fn)
        assert isinstance(result, QueuedResult)
        assert result.result == "hello"

    async def test_measures_execution_seconds(self):
        q = BoundedRequestQueue(max_concurrent=1)

        async def fn():
            await asyncio.sleep(0.05)
            return "done"

        result = await q.submit(fn)
        assert result.execution_seconds >= 0.04

    async def test_uncontended_request_has_near_zero_queue_wait(self):
        q = BoundedRequestQueue(max_concurrent=1)

        async def fn():
            return "ok"

        result = await q.submit(fn)
        assert result.queue_wait_seconds < 0.05

    async def test_propagates_exceptions_from_fn(self):
        q = BoundedRequestQueue(max_concurrent=1)

        async def fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await q.submit(fn)

    async def test_waiting_count_returns_to_zero_after_completion(self):
        q = BoundedRequestQueue(max_concurrent=1)

        async def fn():
            return "ok"

        await q.submit(fn)
        assert q.waiting_count == 0


class TestConcurrencyLimiting:
    async def test_respects_max_concurrent_of_one(self):
        q = BoundedRequestQueue(max_concurrent=1, max_queue_size=5)
        concurrently_running = 0
        max_observed_concurrency = 0

        async def fn():
            nonlocal concurrently_running, max_observed_concurrency
            concurrently_running += 1
            max_observed_concurrency = max(max_observed_concurrency, concurrently_running)
            await asyncio.sleep(0.02)
            concurrently_running -= 1
            return "ok"

        await asyncio.gather(*[q.submit(fn) for _ in range(4)])
        assert max_observed_concurrency == 1

    async def test_allows_up_to_max_concurrent_at_once(self):
        q = BoundedRequestQueue(max_concurrent=3, max_queue_size=10)
        concurrently_running = 0
        max_observed_concurrency = 0

        async def fn():
            nonlocal concurrently_running, max_observed_concurrency
            concurrently_running += 1
            max_observed_concurrency = max(max_observed_concurrency, concurrently_running)
            await asyncio.sleep(0.03)
            concurrently_running -= 1
            return "ok"

        await asyncio.gather(*[q.submit(fn) for _ in range(6)])
        assert max_observed_concurrency == 3

    async def test_later_requests_have_higher_queue_wait_under_contention(self):
        q = BoundedRequestQueue(max_concurrent=1, max_queue_size=5)

        async def fn():
            await asyncio.sleep(0.05)
            return "ok"

        results = await asyncio.gather(*[q.submit(fn) for _ in range(3)])
        waits = [r.queue_wait_seconds for r in results]
        # First request should wait ~0, later ones should wait for earlier
        # ones to finish - proving queue_wait_seconds actually measures
        # contention, not just noise.
        assert waits[0] < waits[-1]


class TestAdmissionControl:
    async def test_rejects_when_queue_is_full(self):
        q = BoundedRequestQueue(max_concurrent=1, max_queue_size=1)
        release = asyncio.Event()

        async def blocking_fn():
            await release.wait()
            return "ok"

        # First submit occupies the single concurrency slot.
        first_task = asyncio.create_task(q.submit(blocking_fn))
        await asyncio.sleep(0.01)  # let it actually start running

        # Second submit occupies the one waiting slot.
        second_task = asyncio.create_task(q.submit(blocking_fn))
        await asyncio.sleep(0.01)

        # Third submit should be rejected: 1 running + 1 waiting = queue full.
        async def instant_fn():
            return "should not run"

        with pytest.raises(QueueFullError) as exc_info:
            await q.submit(instant_fn)
        assert exc_info.value.max_queue_size == 1

        release.set()
        await first_task
        await second_task

    async def test_zero_max_queue_size_rejects_any_contention(self):
        q = BoundedRequestQueue(max_concurrent=1, max_queue_size=0)
        release = asyncio.Event()

        async def blocking_fn():
            await release.wait()
            return "ok"

        first_task = asyncio.create_task(q.submit(blocking_fn))
        await asyncio.sleep(0.01)

        async def instant_fn():
            return "rejected"

        with pytest.raises(QueueFullError):
            await q.submit(instant_fn)

        release.set()
        await first_task

    async def test_queue_full_error_message_is_informative(self):
        err = QueueFullError(waiting=3, max_queue_size=3)
        assert "3" in str(err)


class TestRunningCountStaysAccurate:
    async def test_running_count_accounts_for_requests_that_had_to_wait(self):
        # Regression test: a request that had to wait for a slot (rather
        # than getting one immediately) must still be counted as "running"
        # once it starts, or later admission decisions undercount real
        # concurrency and admit more than max_concurrent.
        q = BoundedRequestQueue(max_concurrent=1, max_queue_size=5)
        release_first = asyncio.Event()
        second_running_count_seen = None

        async def first_fn():
            await release_first.wait()
            return "first"

        async def second_fn():
            nonlocal second_running_count_seen
            second_running_count_seen = q.running_count
            return "second"

        first_task = asyncio.create_task(q.submit(first_fn))
        await asyncio.sleep(0.01)  # let first_fn actually start and occupy the slot
        second_task = asyncio.create_task(q.submit(second_fn))  # must wait
        await asyncio.sleep(0.01)
        assert q.waiting_count == 1  # second is waiting, not yet running

        release_first.set()
        await first_task
        await second_task

        # While second_fn ran, running_count must have been exactly 1 (itself),
        # never 2 - proving the wait->run transition correctly re-incremented
        # _running rather than leaving it stuck at 0.
        assert second_running_count_seen == 1
        assert q.running_count == 0  # back to idle after both complete

    async def test_third_request_correctly_rejected_after_a_wait_run_cycle(self):
        # If _running were left undercounted after the fix above, a third
        # concurrent request could be wrongly admitted as "a free slot"
        # when in fact max_concurrent was already met.
        q = BoundedRequestQueue(max_concurrent=1, max_queue_size=1)
        release = asyncio.Event()

        async def blocking_fn():
            await release.wait()
            return "ok"

        first_task = asyncio.create_task(q.submit(blocking_fn))
        await asyncio.sleep(0.01)
        second_task = asyncio.create_task(q.submit(blocking_fn))
        await asyncio.sleep(0.01)

        async def instant_fn():
            return "should be rejected"

        with pytest.raises(QueueFullError):
            await q.submit(instant_fn)

        release.set()
        await first_task
        await second_task
