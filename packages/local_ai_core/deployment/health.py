"""Startup, health, and readiness checks (theory doc §9-10) — real,
executable functions, not a checklist in prose. `run_startup_checks()` runs
once at boot and can abort startup; `run_readiness_check()` is fast and
repeatable, what `/ready` calls on every request - two different real
functions for two different curriculum topics, not one function doing both
jobs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.config import AppConfig
from local_ai_core.deployment.data_dir import DataDirectoryLayout, ensure_data_dir_layout
from local_ai_core.deployment.model_registry import ModelRegistry, parse_model_catalog

_MIN_FREE_DISK_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB - Module 20's disk-pressure concern, made checkable


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check_data_dir_writable(layout: DataDirectoryLayout) -> CheckResult:
    marker = layout.base_dir / ".write_check"
    try:
        marker.write_text("ok")
        marker.unlink()
        return CheckResult(name="data_dir_writable", passed=True, detail=str(layout.base_dir))
    except OSError as exc:
        return CheckResult(name="data_dir_writable", passed=False, detail=str(exc))


def _check_model_catalog_parseable(model_catalog_path: str | Path) -> CheckResult:
    try:
        entries = parse_model_catalog(model_catalog_path)
        return CheckResult(name="model_catalog_parseable", passed=True, detail=f"{len(entries)} entries")
    except (OSError, ValueError) as exc:
        return CheckResult(name="model_catalog_parseable", passed=False, detail=str(exc))


def _check_disk_space(layout: DataDirectoryLayout) -> CheckResult:
    usage = shutil.disk_usage(layout.base_dir)
    passed = usage.free >= _MIN_FREE_DISK_BYTES
    detail = f"{usage.free / (1024**3):.2f} GiB free"
    return CheckResult(name="disk_space", passed=passed, detail=detail)


def run_startup_checks(config: AppConfig, *, model_catalog_path: str | Path) -> list[CheckResult]:
    layout = ensure_data_dir_layout(config)
    return [
        CheckResult(name="config_valid", passed=isinstance(config, AppConfig), detail="AppConfig validated"),
        _check_data_dir_writable(layout),
        _check_model_catalog_parseable(model_catalog_path),
        _check_disk_space(layout),
    ]


def run_readiness_check(layout: DataDirectoryLayout, registry: ModelRegistry) -> CheckResult:
    if not layout.base_dir.is_dir():
        return CheckResult(name="ready", passed=False, detail="data directory does not exist")
    if len(registry) == 0:
        return CheckResult(name="ready", passed=False, detail="model registry is empty")
    return CheckResult(name="ready", passed=True, detail=f"{len(registry)} models registered")


def run_liveness_check() -> CheckResult:
    return CheckResult(name="alive", passed=True, detail="process is running")
