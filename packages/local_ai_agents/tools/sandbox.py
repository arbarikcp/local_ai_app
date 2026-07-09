"""Path containment (theory doc's tool execution rule: "Can the tool
access this path?") - shared by `file_search.py` and `write_file.py`.
`relative_path` is always resolved *relative to* `allowed_base`, never to
the process's current working directory, so a caller can't sidestep the
sandbox by controlling what directory the tool happens to be run from.
"""

from __future__ import annotations

from pathlib import Path


class PathTraversalError(Exception):
    pass


def resolve_within_sandbox(allowed_base: Path, relative_path: str) -> Path:
    """Raises `PathTraversalError` if `relative_path` (joined to
    `allowed_base`) resolves outside `allowed_base` - a `..`-style escape,
    an absolute path override, or a symlink pointing outside the sandbox.
    """
    base = allowed_base.resolve()
    candidate = (base / relative_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise PathTraversalError(f"{relative_path!r} resolves outside the allowed sandbox {base}")
    return candidate
