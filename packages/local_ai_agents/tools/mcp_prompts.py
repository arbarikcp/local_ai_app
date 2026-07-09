"""Prompts (theory doc §3) - named, parameterized templates a client can
list and render with arguments. Real string formatting and real
missing-argument errors, not a description of the concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PromptNotFoundError(Exception):
    pass


class PromptArgumentError(Exception):
    pass


@dataclass(frozen=True)
class PromptDescriptor:
    name: str
    description: str
    argument_names: list[str]


@dataclass(frozen=True)
class _RegisteredPrompt:
    template: str
    description: str
    argument_names: list[str] = field(default_factory=list)


class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: dict[str, _RegisteredPrompt] = {}

    def register(self, name: str, template: str, description: str, argument_names: list[str] | None = None) -> None:
        self._prompts[name] = _RegisteredPrompt(
            template=template, description=description, argument_names=argument_names or []
        )

    def list(self) -> list[PromptDescriptor]:
        return [
            PromptDescriptor(name=name, description=p.description, argument_names=p.argument_names)
            for name, p in self._prompts.items()
        ]

    def get(self, name: str, arguments: dict[str, str] | None = None) -> str:
        if name not in self._prompts:
            raise PromptNotFoundError(f"No prompt registered with name '{name}'")
        prompt = self._prompts[name]
        arguments = arguments or {}
        missing = [arg for arg in prompt.argument_names if arg not in arguments]
        if missing:
            raise PromptArgumentError(f"Prompt '{name}' is missing required argument(s): {missing}")
        try:
            return prompt.template.format(**arguments)
        except KeyError as exc:
            raise PromptArgumentError(f"Prompt '{name}' references an undeclared argument: {exc}") from exc
