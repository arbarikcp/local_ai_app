import pytest

from local_ai_core.runtimes.errors import (
    ContextTooLarge,
    FeatureNotSupported,
    InvalidModelResponse,
    LLMError,
    ModelNotLoaded,
    ModelOutOfMemory,
    RequestTimeout,
    RuntimeUnavailable,
    SafetyPolicyViolation,
    SchemaValidationError,
    ToolCallValidationError,
)

ALL_SUBCLASSES = [
    RuntimeUnavailable,
    ModelNotLoaded,
    ModelOutOfMemory,
    RequestTimeout,
    InvalidModelResponse,
    SchemaValidationError,
    ToolCallValidationError,
    SafetyPolicyViolation,
    ContextTooLarge,
    FeatureNotSupported,
]


@pytest.mark.parametrize("error_cls", ALL_SUBCLASSES)
def test_every_error_is_an_llm_error(error_cls):
    assert issubclass(error_cls, LLMError)


@pytest.mark.parametrize("error_cls", ALL_SUBCLASSES)
def test_every_error_is_catchable_as_llm_error(error_cls):
    with pytest.raises(LLMError):
        raise error_cls("something went wrong")


def test_llm_error_preserves_message():
    err = RuntimeUnavailable("could not connect")
    assert str(err) == "could not connect"


def test_llm_error_preserves_cause_without_leaking_it_into_the_type():
    original = ValueError("underlying httpx failure")
    err = RequestTimeout("timed out", cause=original)
    assert err.cause is original
    assert not isinstance(err, ValueError)


def test_llm_error_cause_defaults_to_none():
    err = FeatureNotSupported("grammar not supported")
    assert err.cause is None


def test_error_types_are_distinguishable_for_selective_catching():
    # A caller should be able to catch a *specific* error type without also
    # catching unrelated ones - this is the whole point of a taxonomy over a
    # single generic exception.
    try:
        raise RuntimeUnavailable("network down")
    except RequestTimeout:
        pytest.fail("RuntimeUnavailable must not be caught by an except RequestTimeout clause")
    except RuntimeUnavailable:
        pass  # correctly caught as the specific type it was raised as
