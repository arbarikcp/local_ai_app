"""ApprovalGate (theory doc §9) - dangerous tools require an injected
async approval callback to return True before execution. Fails closed:
`NullApprovalGate` (the safe default) always denies, so a dangerous tool
call from a caller that never wired up real approval fails rather than
silently running.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


class ApprovalGate(Protocol):
    async def request_approval(self, tool_name: str, arguments: dict[str, Any]) -> bool: ...


class NullApprovalGate:
    """Always denies - fail closed for a caller that hasn't wired up real
    human approval, rather than a dangerous tool silently running.
    """

    async def request_approval(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return False


class CallbackApprovalGate:
    """Wraps an injected async callback - real usage plugs in something
    that actually prompts a human (a CLI confirmation, a Slack approval
    button, ...); tests inject a fake callback.
    """

    def __init__(self, callback: ApprovalCallback) -> None:
        self._callback = callback

    async def request_approval(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return await self._callback(tool_name, arguments)


class AutoApprovalGate:
    """Always approves. Tests only - explicitly unsafe for real use, since
    it defeats the entire point of a human approval gate.
    """

    async def request_approval(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return True
