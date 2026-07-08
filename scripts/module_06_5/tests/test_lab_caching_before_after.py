import pytest

import lab_caching_before_after as sut
from local_ai_core.gateway.cache import ResponseCache
from local_ai_core.runtimes.fake import FakeRuntime


class TestBuildRepeatedQueryWorkload:
    def test_produces_repeats_times_unique_queries_total_requests(self):
        workload = sut.build_repeated_query_workload(["q1", "q2", "q3"], repeats=4)
        assert len(workload) == 12

    def test_each_unique_query_appears_repeats_times(self):
        workload = sut.build_repeated_query_workload(["q1", "q2"], repeats=3)
        prompts = [r.prompt for r in workload]
        assert prompts.count("q1") == 3
        assert prompts.count("q2") == 3

    def test_uses_the_given_model(self):
        workload = sut.build_repeated_query_workload(["q1"], repeats=1, model="my-model")
        assert workload[0].model == "my-model"


class TestRunWorkloadWithoutCache:
    async def test_calls_runtime_once_per_request_even_for_duplicates(self):
        runtime = FakeRuntime(simulated_latency_ms=0.0)
        workload = sut.build_repeated_query_workload(["q1", "q2"], repeats=3)
        await sut.run_workload_without_cache(runtime, workload)
        assert runtime.call_count == 6  # no caching - every request hits the runtime


class TestRunWorkloadWithCache:
    async def test_calls_runtime_only_once_per_unique_query(self):
        runtime = FakeRuntime(simulated_latency_ms=0.0)
        cache = ResponseCache()
        workload = sut.build_repeated_query_workload(["q1", "q2"], repeats=5)
        await sut.run_workload_with_cache(runtime, cache, workload)
        assert runtime.call_count == 2  # only the 2 unique queries actually generate

    async def test_cache_hit_rate_reflects_the_repetition_ratio(self):
        runtime = FakeRuntime(simulated_latency_ms=0.0)
        cache = ResponseCache()
        # 5 unique queries x 4 repeats = 20 requests, 5 misses + 15 hits.
        workload = sut.build_repeated_query_workload([f"q{i}" for i in range(5)], repeats=4)
        await sut.run_workload_with_cache(runtime, cache, workload)
        assert cache.hit_rate == pytest.approx(15 / 20)


class TestRunLab:
    async def test_with_cache_is_never_slower_than_without_cache(self):
        # With simulated latency > 0, caching must save real (simulated) time -
        # this is the actual "before/after" proof the lab exists to produce.
        results = await sut.run_lab(unique_query_count=3, repeats=4, simulated_latency_ms=5.0)
        assert results["with_cache_seconds"] < results["without_cache_seconds"]
        assert results["speedup"] > 1.0

    async def test_runtime_call_counts_reflect_caching(self):
        results = await sut.run_lab(unique_query_count=4, repeats=3, simulated_latency_ms=0.0)
        assert results["runtime_calls_without_cache"] == 12  # 4 * 3, no caching
        assert results["runtime_calls_with_cache"] == 4  # only unique queries

    async def test_workload_size_and_hit_rate_are_consistent(self):
        results = await sut.run_lab(unique_query_count=5, repeats=4, simulated_latency_ms=0.0)
        assert results["workload_size"] == 20
        assert results["cache_hit_rate"] == pytest.approx(0.75)


class TestResultsToMarkdown:
    def test_renders_all_fields(self):
        results = {
            "workload_size": 20,
            "unique_queries": 5,
            "without_cache_seconds": 0.5,
            "with_cache_seconds": 0.1,
            "speedup": 5.0,
            "cache_hit_rate": 0.75,
            "runtime_calls_without_cache": 20,
            "runtime_calls_with_cache": 5,
        }
        md = sut.results_to_markdown(results)
        assert "20 requests" in md
        assert "5 unique queries" in md
        assert "5.00x" in md
        assert "75%" in md
