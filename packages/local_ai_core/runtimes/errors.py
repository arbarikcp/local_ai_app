"""The canonical LLM error taxonomy (curriculum.md §16).

Every LLMRuntime adapter's public methods raise only these types - never a
runtime-specific exception (httpx.HTTPError, an mlx_lm exception, etc.).
"Normalize errors at the adapter boundary" (theory doc Gotchas) is not
optional: application code written against LLMRuntime must be able to catch
LLMError (or a specific subclass) regardless of which adapter is behind it.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every error an LLMRuntime adapter can raise.

    ``cause`` preserves the original runtime-specific exception (if any) for
    logging/debugging, without leaking its type into calling code's except
    clauses.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class RuntimeUnavailable(LLMError):
    """The runtime could not be reached at all (connection refused, DNS, etc.)."""


class ModelNotLoaded(LLMError):
    """The runtime is reachable, but the requested model isn't available."""


class ModelOutOfMemory(LLMError):
    """The runtime signaled an out-of-memory condition."""


class RequestTimeout(LLMError):
    """The request exceeded its timeout."""


class InvalidModelResponse(LLMError):
    """The runtime returned something the adapter could not parse."""


class SchemaValidationError(LLMError):
    """A response_format was requested but the model's output violates it."""


class ToolCallValidationError(LLMError):
    """A proposed tool call failed validation (Module 14's territory; declared
    here so the taxonomy is complete rather than extended piecemeal later).
    """


class SafetyPolicyViolation(LLMError):
    """A request or response violated a safety policy (Module 22's territory;
    declared here for the same reason as ToolCallValidationError.
    """


class ContextTooLarge(LLMError):
    """The prompt (plus history) exceeds the runtime's context window."""


class FeatureNotSupported(LLMError):
    """The adapter cannot honor a requested response_format/capability.

    Adapters must raise this instead of silently degrading to free-form
    text (theory doc §5) - a caller that asked for constrained output must
    never receive unconstrained output without knowing it.
    """
