"""Resources (theory doc §2) - addressable, read-only content behind
Module 14's `sandbox.py` path containment. A resource URI is always a
sandboxed relative path, never an unconstrained filesystem read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ai_agents.tools.sandbox import resolve_within_sandbox


class ResourceNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class ResourceDescriptor:
    uri: str
    description: str


@dataclass(frozen=True)
class ResourceContent:
    uri: str
    text: str


class ResourceRegistry:
    def __init__(self, allowed_base: Path) -> None:
        self._allowed_base = allowed_base
        self._descriptors: dict[str, ResourceDescriptor] = {}

    def register(self, uri: str, description: str) -> None:
        """`uri` is registered as a real, sandboxed relative path -
        registering something outside the sandbox fails immediately
        rather than deferring the failure to read time.
        """
        resolve_within_sandbox(self._allowed_base, uri)
        self._descriptors[uri] = ResourceDescriptor(uri=uri, description=description)

    def list(self) -> list[ResourceDescriptor]:
        return list(self._descriptors.values())

    def read(self, uri: str) -> ResourceContent:
        if uri not in self._descriptors:
            raise ResourceNotFoundError(f"No resource registered for uri '{uri}'")
        path = resolve_within_sandbox(self._allowed_base, uri)
        text = path.read_text(encoding="utf-8")
        return ResourceContent(uri=uri, text=text)
