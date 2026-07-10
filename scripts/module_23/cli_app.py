"""Lab 1 - package CLI. A real `typer.Typer()` app - this repo's first
actual use of the `typer` dependency, declared since Phase 0 but never
wired to anything until now. Every command drives real Module 23 package
code (config loading, the composition root, startup checks, backup/restore)
- nothing here is a placeholder.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

import typer  # noqa: E402

from local_ai_core.deployment.app_context import build_app_context  # noqa: E402
from local_ai_core.deployment.backup import backup_sqlite_db, list_backups, restore_sqlite_db  # noqa: E402
from local_ai_core.deployment.config import load_config  # noqa: E402
from local_ai_core.deployment.health import run_startup_checks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"

app = typer.Typer(help="Local AI app CLI (Module 23).")


@app.command()
def check(config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    """Run startup checks and print a pass/fail report."""
    config = load_config(config_path)
    results = run_startup_checks(config, model_catalog_path=DEFAULT_CATALOG_PATH)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        typer.echo(f"[{status}] {result.name}: {result.detail}")
    if not all(r.passed for r in results):
        raise typer.Exit(code=1)


@app.command()
def models(config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    """List the registered models from the real model catalog."""
    config = load_config(config_path)
    ctx = build_app_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)
    for entry in ctx.model_registry.all_entries():
        typer.echo(f"{entry.model_id} ({entry.category}, {entry.recommended_ram_tier})")


@app.command()
def backup(config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    """Back up the audit log database."""
    config = load_config(config_path)
    ctx = build_app_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)
    backup_path = backup_sqlite_db(ctx.data_dir.audit_db, ctx.data_dir.backups_dir)
    typer.echo(f"Backed up to {backup_path}")


@app.command()
def restore(backup_path: str, config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    """Restore the audit log database from a backup file."""
    config = load_config(config_path)
    ctx = build_app_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)
    restore_sqlite_db(backup_path, ctx.data_dir.audit_db)
    typer.echo(f"Restored {ctx.data_dir.audit_db} from {backup_path}")


@app.command()
def list_backup_files(config_path: str = str(DEFAULT_CONFIG_PATH)) -> None:
    """List available backup files."""
    config = load_config(config_path)
    ctx = build_app_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)
    for backup_file in list_backups(ctx.data_dir.backups_dir):
        typer.echo(str(backup_file))


@app.command()
def serve(config_path: str = str(DEFAULT_CONFIG_PATH), port: int = 8000) -> None:
    """Print the command to start the local API service (doesn't block the CLI itself)."""
    typer.echo(f"uv run uvicorn api_app:app --port {port} --app-dir scripts/module_23")
    typer.echo(f"(using config: {config_path})")


if __name__ == "__main__":
    app()
