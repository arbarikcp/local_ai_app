import pytest

from local_ai_core.optimization.benchmark_harness import BenchmarkConfig, run_benchmark
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest

REQUEST = LLMRequest(model="test-model", prompt="hello there")


class TestRunBenchmark:
    async def test_measures_real_simulated_latency(self):
        fast = FakeRuntime(default_response="hi", simulated_latency_ms=5)
        slow = FakeRuntime(default_response="hi", simulated_latency_ms=40)
        configs = [
            BenchmarkConfig(name="fast", runtime=fast, request=REQUEST),
            BenchmarkConfig(name="slow", runtime=slow, request=REQUEST),
        ]

        results = await run_benchmark(configs, repeats=2)

        by_name = {r.name: r for r in results}
        assert by_name["fast"].mean_latency_ms < by_name["slow"].mean_latency_ms
        assert by_name["fast"].sample_count == 2
        assert by_name["slow"].sample_count == 2

    async def test_p95_is_at_least_the_mean_for_a_uniform_distribution(self):
        runtime = FakeRuntime(default_response="hi", simulated_latency_ms=5)
        configs = [BenchmarkConfig(name="uniform", runtime=runtime, request=REQUEST)]

        results = await run_benchmark(configs, repeats=5)

        assert results[0].p95_latency_ms >= results[0].mean_latency_ms * 0.9

    async def test_tokens_per_second_is_positive_for_a_nonempty_response(self):
        runtime = FakeRuntime(default_response="one two three four five", simulated_latency_ms=10)
        configs = [BenchmarkConfig(name="config", runtime=runtime, request=REQUEST)]

        results = await run_benchmark(configs, repeats=1)

        assert results[0].mean_tokens_per_second > 0

    async def test_calls_the_runtime_repeats_times(self):
        runtime = FakeRuntime(default_response="hi")
        configs = [BenchmarkConfig(name="config", runtime=runtime, request=REQUEST)]

        await run_benchmark(configs, repeats=4)

        assert runtime.call_count == 4

    async def test_nonpositive_repeats_raises(self):
        runtime = FakeRuntime(default_response="hi")
        configs = [BenchmarkConfig(name="config", runtime=runtime, request=REQUEST)]

        with pytest.raises(ValueError):
            await run_benchmark(configs, repeats=0)
