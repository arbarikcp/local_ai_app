"""Tool, ToolResult, and the tool error taxonomy (theory doc §2, §6-7) - a
tool pairs a Pydantic args schema with a handler function, and every
handler call goes through the same uniform result/error shape regardless
of what the tool actually does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Type

from pydantic import BaseModel


class ToolError(Exception):
    """Base class for every tool-related failure - mirrors Module 6's
    LLMError hierarchy so callers can catch broadly or narrowly.
    """


class ToolNotFoundError(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolPermissionError(ToolError):
    pass


class ToolApprovalRequiredError(ToolError):
    pass


class ToolBudgetExceededError(ToolError):
    pass


class ToolExecutionError(ToolError):
    """The tool's own handler raised - wrapped so a bug inside a handler
    never looks identical to a validation or permission failure.
    """


@dataclass(frozen=True)
class ToolResult:
    success: bool
    data: Any = None
    error_message: str | None = None

    def as_text(self) -> str:
        """A uniform text representation for feeding back into a
        conversation - callers never need to special-case each tool's own
        return shape.
        """
        if self.success:
            return str(self.data)
        return f"Error: {self.error_message}"


HandlerFn = Callable[[BaseModel], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: HandlerFn
    dangerous: bool = False

    def json_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "dangerous": self.dangerous,
            "parameters": self.args_model.model_json_schema(),
        }


@dataclass(frozen=True)
class ToolCallProposal:
    """What the LLM decides (theory doc §1) - inert data, never executed
    directly. `raw_arguments` is the un-validated dict from the model's
    response; `ToolExecutor` is the only thing that turns this into a
    validated `args_model` instance.
    """

    tool_name: str
    raw_arguments: dict[str, Any] = field(default_factory=dict)
