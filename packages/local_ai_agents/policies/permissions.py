"""PermissionPolicy (theory doc §8) - real role-based allow/deny, checked
before a tool ever runs. A role either can or cannot call a given tool
name; nothing about what the model says it's allowed to do is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_WILDCARD = "*"


@dataclass
class PermissionPolicy:
    allowed_tools_by_role: dict[str, set[str]] = field(default_factory=dict)

    def allow(self, role: str, tool_name: str) -> None:
        self.allowed_tools_by_role.setdefault(role, set()).add(tool_name)

    def allow_all(self, role: str) -> None:
        """Grants every tool name to `role`, present and future - an
        explicit, auditable wildcard rather than an accidental one.
        """
        self.allowed_tools_by_role.setdefault(role, set()).add(_WILDCARD)

    def is_allowed(self, role: str, tool_name: str) -> bool:
        granted = self.allowed_tools_by_role.get(role, set())
        return tool_name in granted or _WILDCARD in granted
