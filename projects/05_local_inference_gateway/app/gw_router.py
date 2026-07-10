"""Task-based routing (ARCHITECTURE.md "Task routing") — the confirmed
gap PROPOSAL.md's survey found: nothing in the repo maps a task string to
a primary/fallback model pair. `load_routes()` validates every route's
model_id against the real, injected `ModelRegistry` at load time — an
unknown model_id in the config is a startup error, not a first-request
surprise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from local_ai_core.deployment.model_registry import ModelRegistry


class UnknownModelInRouteError(ValueError):
    """A route names a model_id that isn't present in the injected ModelRegistry."""


class TaskNotFoundError(KeyError):
    """A caller asked for a task name that has no configured route."""


@dataclass(frozen=True)
class TaskRoute:
    task: str
    primary_model: str
    fallback_model: str
    max_context_tokens: int


def load_routes(path: str | Path, *, model_registry: ModelRegistry) -> dict[str, TaskRoute]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    routes: dict[str, TaskRoute] = {}
    for task, entry in raw["routes"].items():
        primary_model = entry["primary"]
        fallback_model = entry["fallback"]
        for model_id in (primary_model, fallback_model):
            if model_registry.get(model_id) is None:
                raise UnknownModelInRouteError(
                    f"route {task!r} names model_id {model_id!r}, which is not in the model registry"
                )
        routes[task] = TaskRoute(
            task=task,
            primary_model=primary_model,
            fallback_model=fallback_model,
            max_context_tokens=entry["max_context_tokens"],
        )
    return routes


def resolve_route(routes: dict[str, TaskRoute], task: str) -> TaskRoute:
    try:
        return routes[task]
    except KeyError as exc:
        raise TaskNotFoundError(f"no route configured for task {task!r}") from exc
