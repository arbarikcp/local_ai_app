"""PromptRegistry — versioned storage of named prompt templates (theory doc
§10). Every named prompt has a version; this is not bookkeeping for its own
sake - Module 6.5's response_cache_key() takes prompt_version as a required
parameter specifically so a prompt edit invalidates cached responses
instead of silently serving stale output under a changed prompt.
"""

from __future__ import annotations

from .template import PromptTemplate


class PromptNotFoundError(KeyError):
    def __init__(self, prompt_id: str, version: str | None = None) -> None:
        if version is None:
            super().__init__(f"No prompt registered under id {prompt_id!r}")
        else:
            super().__init__(f"No version {version!r} registered for prompt id {prompt_id!r}")
        self.prompt_id = prompt_id
        self.version = version


class PromptRegistry:
    """In-memory versioned registry. Register once per (prompt_id, version);
    re-registering the same pair is rejected to keep a version's content
    immutable once published - edit by publishing a new version instead.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, PromptTemplate]] = {}
        self._latest: dict[str, str] = {}

    def register(self, template: PromptTemplate) -> None:
        versions = self._store.setdefault(template.prompt_id, {})
        if template.version in versions:
            raise ValueError(
                f"Prompt {template.prompt_id!r} version {template.version!r} is already "
                "registered - versions are immutable once published, register a new version instead"
            )
        versions[template.version] = template
        self._latest[template.prompt_id] = template.version

    def get(self, prompt_id: str, version: str | None = None) -> PromptTemplate:
        """``version=None`` returns the most-recently-registered version."""
        versions = self._store.get(prompt_id)
        if versions is None:
            raise PromptNotFoundError(prompt_id)
        resolved_version = version or self._latest[prompt_id]
        template = versions.get(resolved_version)
        if template is None:
            raise PromptNotFoundError(prompt_id, resolved_version)
        return template

    def list_versions(self, prompt_id: str) -> list[str]:
        versions = self._store.get(prompt_id)
        if versions is None:
            raise PromptNotFoundError(prompt_id)
        return sorted(versions.keys())

    def list_prompt_ids(self) -> list[str]:
        return sorted(self._store.keys())

    def latest_version(self, prompt_id: str) -> str:
        if prompt_id not in self._latest:
            raise PromptNotFoundError(prompt_id)
        return self._latest[prompt_id]
