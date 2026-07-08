"""Lab 2.1 support — verify the base dev-tool environment is present.

Read-only: this module never installs, downloads, or modifies anything. It
shells out to `--version`/`-n` style flags only, so it is safe to run on any
machine, including one where no model runtime should be installed (see
docs/modules/02_mac_local_ai_development_environment.md).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass

REQUIRED_TOOLS = ["git", "make", "cmake", "uv", "jq", "rg"]
OPTIONAL_TOOLS = ["ollama", "brew"]

# Homebrew formula names differ from binary names for a couple of these.
_BREW_FORMULA_FOR_BINARY = {"rg": "ripgrep"}


def _brew_formula(binary_name: str) -> str:
    return _BREW_FORMULA_FOR_BINARY.get(binary_name, binary_name)


@dataclass(frozen=True)
class ToolCheck:
    name: str
    found: bool
    path: str | None
    version: str | None
    required: bool


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def check_tool(name: str, required: bool, version_flag: str = "--version") -> ToolCheck:
    path = shutil.which(name)
    if path is None:
        return ToolCheck(name=name, found=False, path=None, version=None, required=required)
    try:
        result = subprocess.run(
            [path, version_flag], capture_output=True, text=True, timeout=5, check=False
        )
        version = _first_line(result.stdout or result.stderr)
    except (subprocess.TimeoutExpired, OSError):
        version = None
    return ToolCheck(name=name, found=True, path=path, version=version, required=required)


def check_all_tools() -> list[ToolCheck]:
    checks = [check_tool(t, required=True) for t in REQUIRED_TOOLS]
    checks += [check_tool(t, required=False) for t in OPTIONAL_TOOLS]
    return checks


@dataclass(frozen=True)
class MachineProfile:
    architecture: str  # "arm64" (Apple Silicon) or "x86_64" (Intel) etc.
    is_apple_silicon: bool
    macos_version: str
    python_version: str


def detect_machine_profile() -> MachineProfile:
    arch = platform.machine()
    return MachineProfile(
        architecture=arch,
        is_apple_silicon=(arch == "arm64"),
        macos_version=platform.mac_ver()[0] or "unknown",
        python_version=platform.python_version(),
    )


def missing_required_tools(checks: list[ToolCheck]) -> list[str]:
    return [c.name for c in checks if c.required and not c.found]


def tools_to_markdown_table(checks: list[ToolCheck]) -> str:
    header = "| Tool | Required | Found | Path | Version |\n|---|---|---|---|---|\n"
    lines = []
    for c in checks:
        lines.append(
            f"| {c.name} | {'yes' if c.required else 'no'} | "
            f"{'yes' if c.found else 'no'} | {c.path or 'n/a'} | {c.version or 'n/a'} |"
        )
    return header + "\n".join(lines)


def main() -> int:
    profile = detect_machine_profile()
    print(
        f"# Machine profile\n\n"
        f"- Architecture: {profile.architecture} "
        f"({'Apple Silicon' if profile.is_apple_silicon else 'not Apple Silicon — MLX unavailable'})\n"
        f"- macOS version: {profile.macos_version}\n"
        f"- Python version: {profile.python_version}\n"
    )
    checks = check_all_tools()
    print("# Dev tool check\n")
    print(tools_to_markdown_table(checks))
    missing = missing_required_tools(checks)
    if missing:
        print(f"\nMISSING required tools: {', '.join(missing)}")
        formulas = " ".join(_brew_formula(t) for t in missing)
        print(f"Install with: brew install {formulas}")
        return 1
    print("\nAll required tools present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
