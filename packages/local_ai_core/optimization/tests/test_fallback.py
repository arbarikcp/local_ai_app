import pytest

from local_ai_core.optimization.fallback import FallbackRuntime, NoRuntimesAvailable
from local_ai_core.runtimes.errors import RuntimeUnavailable, SchemaValidationError
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest

REQUEST = LLMRequest(model="test-model", prompt="hello")


class TestPrimaryRuntimeSucceeds:
    async def test_uses_the_first_runtime_when_it_succeeds(self):
        primary = FakeRuntime(default_response="primary answer")
        secondary = FakeRuntime(default_response="secondary answer")
        chain = FallbackRuntime([primary, secondary])

        result = await chain.generate(REQUEST)

        assert result.response.text == "primary answer"
        assert result.runtime_index == 0
        assert result.attempts == 1
        assert secondary.call_count == 0


class TestFallsBackOnRetryableError:
    async def test_falls_back_to_the_second_runtime_on_runtime_unavailable(self):
        primary = FakeRuntime(fail_with=RuntimeUnavailable("primary is down"))
        secondary = FakeRuntime(default_response="secondary answer")
        chain = FallbackRuntime([primary, secondary])

        result = await chain.generate(REQUEST)

        assert result.response.text == "secondary answer"
        assert result.runtime_index == 1
        assert result.attempts == 2

    async def test_falls_through_multiple_failing_runtimes(self):
        first = FakeRuntime(fail_with=RuntimeUnavailable("down"))
        second = FakeRuntime(fail_with=RuntimeUnavailable("also down"))
        third = FakeRuntime(default_response="third answer")
        chain = FallbackRuntime([first, second, third])

        result = await chain.generate(REQUEST)

        assert result.response.text == "third answer"
        assert result.runtime_index == 2
        assert result.attempts == 3


class TestAllRuntimesFail:
    async def test_raises_no_runtimes_available_when_every_runtime_fails(self):
        first = FakeRuntime(fail_with=RuntimeUnavailable("down"))
        second = FakeRuntime(fail_with=RuntimeUnavailable("also down"))
        chain = FallbackRuntime([first, second])

        with pytest.raises(NoRuntimesAvailable):
            await chain.generate(REQUEST)


class TestNonRetryableErrorsPropagateImmediately:
    async def test_a_validation_failure_does_not_trigger_fallback(self):
        primary = FakeRuntime(fail_with=SchemaValidationError("bad schema"))
        secondary = FakeRuntime(default_response="should never be called")
        chain = FallbackRuntime([primary, secondary])

        with pytest.raises(SchemaValidationError):
            await chain.generate(REQUEST)

        assert secondary.call_count == 0


class TestConstructorValidation:
    def test_empty_runtime_list_raises(self):
        with pytest.raises(ValueError):
            FallbackRuntime([])
