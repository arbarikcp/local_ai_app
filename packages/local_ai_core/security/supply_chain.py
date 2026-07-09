"""Model supply-chain verification (theory doc §8) — real SHA-256 checksum
computation and comparison against a trusted manifest entry. Same
content-hashing discipline Module 19's `hash_dataset()` and
`local_ai_rag`'s incremental indexer already established for datasets,
applied here to model files - a real, checkable fact about a file, not a
claim about where it came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelManifestEntry:
    name: str
    source_url: str
    sha256: str
    license: str


class ChecksumMismatchError(Exception):
    def __init__(self, name: str, expected: str, actual: str) -> None:
        super().__init__(f"Checksum mismatch for {name!r}: expected {expected}, got {actual}")
        self.name = name
        self.expected = expected
        self.actual = actual


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_against_manifest(path: str | Path, entry: ModelManifestEntry) -> None:
    """Raises ChecksumMismatchError if `path`'s real SHA-256 doesn't match
    `entry.sha256` - silent success otherwise (verification passed).
    """
    actual = compute_sha256(path)
    if actual != entry.sha256:
        raise ChecksumMismatchError(entry.name, entry.sha256, actual)
