"""Labs 5-6 - add queueing and streaming. No new package code: Module
6.5's `BoundedRequestQueue` and Module 6's `FakeRuntime.stream()` already do
this for real - this script wires both together, submitting several
concurrent streaming requests through a bounded queue and proving real
admission behavior (some requests wait, none are silently dropped) plus
real incremental token delivery.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.gateway.queue import BoundedRequestQueue  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402


async def _stream_via_queue(queue: BoundedRequestQueue, runtime: FakeRuntime, request: LLMRequest) -> list[str]:
    async def collect() -> list[str]:
        return [chunk async for chunk in runtime.stream(request)]

    queued_result = await queue.submit(collect)
    return queued_result.result


async def run_lab() -> dict:
    runtime = FakeRuntime(default_response="billing category ticket", simulated_latency_ms=10)
    queue = BoundedRequestQueue(max_concurrent=2, max_queue_size=5)

    requests = [LLMRequest(model="router-demo", prompt=f"ticket {i}") for i in range(4)]

    results = await asyncio.gather(*(_stream_via_queue(queue, runtime, request) for request in requests))

    return {
        "request_count": len(requests),
        "streamed_chunk_counts": [len(r) for r in results],
        "first_request_full_text": "".join(results[0]).strip(),
        "final_running_count": queue.running_count,
        "final_waiting_count": queue.waiting_count,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 5-6 - queueing and streaming (Module 6.5 + Module 6 unchanged)\n\n"
        f"- Ran {result['request_count']} concurrent streaming requests through a "
        f"max_concurrent=2 bounded queue\n"
        f"- Streamed chunk counts per request: {result['streamed_chunk_counts']}\n"
        f"- First request's reassembled text: \"{result['first_request_full_text']}\"\n"
        f"- Queue drained cleanly: running={result['final_running_count']}, "
        f"waiting={result['final_waiting_count']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
