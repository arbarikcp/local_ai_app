"""Model registry (theory doc §4-5) — the first program to ever read
`models/MODEL_CATALOG.md` programmatically. `parse_model_catalog()` extracts
the file's real embedded YAML fences (Module 3's own committed format) and
validates each into a `ModelRegistryEntry` - a real parser against a real,
already-committed file, not a fresh catalog invented for this module.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

_YAML_FENCE_RE = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)

# The real catalog uses a tri-state value here (true/false/"maybe" - "maybe"
# means untested-but-plausible, distinct from a confirmed false) - modeled
# honestly rather than coercing "maybe" into a boolean and losing that
# uncertainty signal.
RuntimeSupportValue = bool | Literal["maybe"]


class RuntimeSupport(BaseModel):
    ollama: RuntimeSupportValue = False
    gguf: RuntimeSupportValue = False
    mlx: RuntimeSupportValue = False


class ModelRegistryEntry(BaseModel):
    model_id: str
    family: str
    category: str
    runtime: RuntimeSupport
    recommended_ram_tier: str
    quantization_tested: list[str] = []
    context_tested: list[int] = []
    use_cases: list[str] = []
    known_issues: list[str] = []
    license_notes: str = ""
    last_verified: date | str = ""
    verification_status: str = ""


def parse_model_catalog(path: str | Path) -> list[ModelRegistryEntry]:
    text = Path(path).read_text(encoding="utf-8")
    entries = []
    for match in _YAML_FENCE_RE.finditer(text):
        raw = yaml.safe_load(match.group(1))
        entries.append(ModelRegistryEntry.model_validate(raw))
    return entries


class ModelRegistry:
    def __init__(self, entries: list[ModelRegistryEntry]) -> None:
        self._entries = entries
        self._by_id = {entry.model_id: entry for entry in entries}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, model_id: str) -> ModelRegistryEntry | None:
        return self._by_id.get(model_id)

    def by_category(self, category: str) -> list[ModelRegistryEntry]:
        return [e for e in self._entries if e.category == category]

    def categories(self) -> set[str]:
        return {e.category for e in self._entries}

    def all_entries(self) -> list[ModelRegistryEntry]:
        return list(self._entries)


def load_model_registry(path: str | Path) -> ModelRegistry:
    return ModelRegistry(parse_model_catalog(path))
