"""Lab 2.1 / theory §10-12 support — locate and size model runtime caches.

Read-only: reports on directories if they already exist. Never creates,
downloads into, or deletes anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheLocation:
    runtime: str
    path: Path
    exists: bool
    size_bytes: int | None  # None when the path doesn't exist


def default_cache_paths(home: Path | None = None) -> dict[str, Path]:
    """The default cache directory for each runtime this module covers.

    ``home`` is injectable for testing; defaults to the real home directory.
    """
    home = home or Path.home()
    return {
        "ollama": home / ".ollama" / "models",
        "huggingface (used by mlx-lm)": home / ".cache" / "huggingface" / "hub",
    }


def directory_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under ``path``. Follows no symlinks out of
    caution around cache directories that may symlink to shared blob stores.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                if not file_path.is_symlink():
                    total += file_path.stat().st_size
            except OSError:
                continue
    return total


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover - unreachable given loop above


def scan_caches(home: Path | None = None) -> list[CacheLocation]:
    results = []
    for runtime, path in default_cache_paths(home).items():
        exists = path.exists()
        size = directory_size_bytes(path) if exists else None
        results.append(CacheLocation(runtime=runtime, path=path, exists=exists, size_bytes=size))
    return results


def caches_to_markdown_table(locations: list[CacheLocation]) -> str:
    header = "| Runtime | Cache path | Exists | Size |\n|---|---|---|---:|\n"
    lines = []
    for loc in locations:
        size_str = format_bytes(loc.size_bytes) if loc.size_bytes is not None else "n/a"
        lines.append(f"| {loc.runtime} | `{loc.path}` | {'yes' if loc.exists else 'no'} | {size_str} |")
    return header + "\n".join(lines)


def main() -> int:
    locations = scan_caches()
    print("# Model cache report\n")
    print(caches_to_markdown_table(locations))
    print(
        "\nNote: llama.cpp / llama-cpp-python GGUF files are not auto-cached — you point "
        "--model / model_path directly at wherever you saved them, so there is no default "
        "path to scan for that runtime."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
