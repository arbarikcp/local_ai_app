# Operations Runbook — Local AI App (Module 23)

> Companion to [docs/modules/23_packaging_and_deployment.md](../modules/23_packaging_and_deployment.md). Curriculum reference: curriculum.md §33, Lab 6.

## Starting the CLI

```bash
uv run python scripts/module_23/cli_app.py check     # run startup checks
uv run python scripts/module_23/cli_app.py models    # list the model registry
uv run python scripts/module_23/cli_app.py serve     # print the command to start the API
```

All commands accept `--config-path <file>` to point at a config other than
`config/app.example.yaml`. **The default config's `app.data_dir` is
`~/.local-llm-ai`** — running `check`/`models`/`backup` with the default config
creates real (small, harmless) directories and an empty SQLite audit log under
your actual home directory, exactly as the config declares. Use a test config
pointing at a scratch directory (see `scripts/module_23/tests/test_cli_app.py`
for the pattern) if you want to avoid touching your real home directory.

## Starting the API service

```bash
uv run uvicorn api_app:app --port 8000 --app-dir scripts/module_23
```

Set `APP_CONFIG_PATH=/path/to/config.yaml` before starting to use a non-default
config. Endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness — is the process up at all |
| `/ready` | GET | Readiness — is the data directory present and the model registry populated |
| `/models` | GET | Lists every entry in the model registry |
| `/chat` | POST | `{"prompt": "...", "model": "optional-override"}` — guarded by the security classifier, admission-controlled, answered by the injected runtime (`FakeRuntime` by default on this machine) |

## Running startup checks

```bash
uv run python scripts/module_23/cli_app.py check
```

Exits non-zero if any check fails. The four checks:

1. **config_valid** — the config file parsed and validated against `AppConfig`.
2. **data_dir_writable** — a real write-then-delete probe against `app.data_dir`.
3. **model_catalog_parseable** — `models/MODEL_CATALOG.md` parsed without error.
4. **disk_space** — at least 1 GiB free at `app.data_dir`'s filesystem (Module 20's
   disk-pressure concern, made checkable).

Run this before starting the API service in any new environment, and after any
config or model-catalog change.

## Backup and restore

```bash
uv run python scripts/module_23/cli_app.py backup
uv run python scripts/module_23/cli_app.py list-backup-files
uv run python scripts/module_23/cli_app.py restore <path-to-backup-file>
```

`backup` uses SQLite's real `.backup()` API (a consistent snapshot even
mid-write, not a naive file copy) against the audit log at
`<data_dir>/audit/audit.db`, writing a timestamped copy to
`<data_dir>/backups/`. `restore` reverses the operation. Back up before any
schema change to `AuditLog`, `SessionStore`, `AdapterRegistry`, or
`EvalFeedbackStore` — all four are plain SQLite files under `<data_dir>/`
and can be backed up the same way by pointing `backup_sqlite_db()` at any of
them directly (the CLI command above only wires up the audit log by default).

## Offline mode

`AppConfig.app.offline_mode` (default `false`). When `true`, no code path in
this repo should attempt a network call — the default `FakeRuntime` never
does regardless of this flag, structurally: it's a pure in-memory object with
no HTTP client at all (not separately tested here, since there's nothing to
call). A real `MLXRuntime`/`OllamaRuntime` injected via
`build_app_context(..., runtime=...)` on the resourced Mac would need its own
offline-mode check before this flag means anything for a real model call.
Document any future real-runtime integration's offline behavior here.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `check` fails on `data_dir_writable` | Permissions issue on `app.data_dir`, or the path is on a read-only volume | Check `ls -la` on the parent directory; fix permissions or change `app.data_dir` |
| `check` fails on `model_catalog_parseable` | `models/MODEL_CATALOG.md` moved, deleted, or has a malformed YAML fence | Confirm the file exists at the expected path; validate each ` ```yaml ` fence parses with `yaml.safe_load` |
| `check` fails on `disk_space` | Less than 1 GiB free at `app.data_dir`'s filesystem | Free disk space or back up + prune old SQLite backups under `<data_dir>/backups/` |
| `/chat` returns 400 | The security guard classifier (`RuleBasedGuardClassifier`) detected a prompt-injection pattern and raised via `SafetyPolicyViolation`-equivalent HTTP handling | Expected behavior for a flagged prompt — check the response `detail` field for which pattern matched; see Module 22's report for the classifier's known false-negative gap |
| `/chat` returns 429 | `QueueFullError` — Module 6.5's `BoundedRequestQueue` is at capacity (`limits.max_concurrent_requests`/queue size exceeded) | Expected under load with `max_concurrent_requests: 1` (the deliberate Mac-local default) — retry with backoff, or raise the limit only after a real concurrency measurement (Module 6.5's `recommend_policy_from_measurements()`) |
| `/ready` returns 503 | Data directory missing or model registry failed to load | Run `check` first; confirm `model_catalog_path` points at a real, parseable file |
| A model call raises `RuntimeUnavailable`/`RequestTimeout` (real runtime, resourced Mac) | The injected runtime (e.g. `MLXRuntime`/`OllamaRuntime`) couldn't reach the model server, or the call exceeded its timeout | Check the runtime server is running (Module 5); see Module 20's `FallbackRuntime` for a real fallback-chain pattern if this becomes frequent |
| A tool call never returns | A hung tool handler with no timeout wrapper | Wrap the handler with Module 22's `security/tool_call_timeout.py::with_timeout()` |

## Versioning

This app's version is the git tag of the commit it was built from
(`git tag module-N` per module, `git describe --tags` for the running
commit). There is no separate semantic-version file to keep in sync.
