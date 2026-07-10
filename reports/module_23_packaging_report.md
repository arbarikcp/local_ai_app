# Module 23 deliverable — packaging and deployment report

Status: **complete.** This is the final module of the 1-23 curriculum arc and the first to build
a real composition root: every prior module's demo script wired its own ad hoc subset of
imports, but nothing before this assembled runtime + gateway admission control + security guard
pipeline + metrics into one running application. Module 23 builds that assembly (`AppContext`),
then a real CLI (`typer`, a dependency declared since Phase 0 and never once used until now) and
a real FastAPI service on top of it. No honest-skip surface beyond the model runtime itself
(`FakeRuntime`, this repo's standing default since Module 6) — config loading, model-registry
parsing, data-directory creation, startup/health/readiness checks, SQLite backup/restore, the
CLI, and the FastAPI service (via `TestClient`) all run for real.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `deployment/config.py` | 5 | Real Pydantic validation of curriculum's exact config shape, loaded from the real committed `config/app.example.yaml` |
| `deployment/model_registry.py` | 14 | Real parsing of all 10 entries in `models/MODEL_CATALOG.md`'s embedded YAML fences, including a genuine tri-state `mlx: true/false/maybe` value |
| `deployment/data_dir.py` | 5 | Real subdirectory creation under `app.data_dir`, idempotent, one path per existing SQLite store |
| `deployment/health.py` | 8 | Four real startup checks including a genuine failure (missing model catalog), plus readiness/liveness |
| `deployment/backup.py` | 6 | Real SQLite `.backup()` API against a real `AuditLog`, a genuine backup-then-restore-then-read round trip |
| `deployment/app_context.py` | 6 | The composition root wiring runtime, admission control, guard classifier, and audit log together |
| `scripts/module_23/cli_app.py` | 6 | A real `typer` CLI: `check`, `models`, `backup`/`restore`/`list-backup-files`, `serve` |
| `scripts/module_23/api_app.py` | 6 | A real FastAPI app tested via `TestClient`: health, readiness, model listing, a guarded chat endpoint |
| `scripts/module_23/config_and_registry_demo.py`, `startup_checks_demo.py` | 6 | Labs 3-5 exercised for real, including a deliberately broken configuration |
| `notebooks/23_packaging_and_deployment.ipynb` | — | **Executed end-to-end** — every cell a real computation, entirely in temporary directories |

**53 new tests this module** (1803 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Real proof: the model registry parses curriculum's own tri-state runtime field honestly

`models/MODEL_CATALOG.md` genuinely uses `mlx: true`, `mlx: false`, and `mlx: maybe` (untested-
but-plausible support, distinct from a confirmed `false`) across its 10 entries. The first
version of `ModelRegistryEntry` coerced this to a plain `bool` and crashed on the real file with
a Pydantic `bool_parsing` error — caught immediately by running the parser against the real
catalog rather than a synthetic fixture, and fixed by modeling `RuntimeSupportValue = bool |
Literal["maybe"]` instead of forcing a lossy boolean coercion.

## Real proof: startup checks catch a real, specific failure

```
## Broken configuration (missing model catalog)
- [PASS] config_valid: AppConfig validated
- [PASS] data_dir_writable: /var/folders/.../data
- [FAIL] model_catalog_parseable: [Errno 2] No such file or directory: '.../does_not_exist.md'
- [PASS] disk_space: 4.85 GiB free

All passed: False
```

`run_startup_checks()` isolates the failure to exactly the broken check — `data_dir_writable`
and `disk_space` still pass against the same real temporary directory, proving the four checks
are independent, not a single pass/fail blob that can't tell an operator which thing is actually
wrong.

## Real proof: the composition root produces a fully wired, working app

```
Runtime: FakeRuntime
Model registry: 10 entries
Admission policy: max_concurrent_requests=1
Data directory: /var/folders/.../data
```

`build_app_context()` is the first function in this repo to construct all of these together from
one `AppConfig`. The CLI's `check`/`models`/`backup` commands and the API's every endpoint all
call it — no separate wiring path for the CLI vs. the API, so a fix or extension to `AppContext`
automatically applies to both surfaces.

## Real proof: the guarded `/chat` endpoint genuinely blocks an injection attempt

```
POST /chat (clean) -> 200 {'text': '(FakeRuntime - no model runtime installed on this machine)', 'model': 'llama3.2:3b'}
POST /chat (injection) -> 400 {'detail': 'request blocked: 2 prompt-injection pattern(s) matched'}
```

The same `RuleBasedGuardClassifier` built in Module 22 runs inside a real ASGI request handler
here — a clean prompt reaches the (fake) runtime and gets a 200, an injection prompt is rejected
with a 400 before the runtime is ever called. This is the security guard pipeline's first real
deployment inside an actual HTTP service, not just a standalone classifier demo.

## Real proof: backup and restore is a genuine file-level round trip

```
Backed up to: /var/folders/.../data/backups/audit_20260710T070416903506Z.db
Restored entries: [AuditEntry(trace_id='trace-1', tool_name='lookup_order', ...)]
```

`backup_sqlite_db()` uses `sqlite3.Connection.backup()` — the correct API for backing up a live
database (a consistent snapshot even mid-write), not a naive file copy that could catch a
half-written page mid-transaction. A real `AuditLog` entry written before the backup is present,
unchanged, in a completely separate SQLite file restored from the backup, read back through a
fresh `AuditLog` instance pointed at the restored path.

## A real side effect, disclosed: the default config writes to the user's actual home directory

Running `cli_app.py check`/`models`/`backup` with the default `config/app.example.yaml`
(matching curriculum's own example, `app.data_dir: ~/.local-llm-ai`) creates real, small,
harmless directories and an empty SQLite audit log under the user's actual home directory — this
is the config's literal, intended behavior for a real packaged local app, not a bug. Every test
and every notebook cell in this module deliberately uses a temporary directory instead (verified
by checking the real `~/.local-llm-ai/audit/audit.db`'s mtime was unchanged before and after the
full test suite and notebook execution), and the operations runbook documents this explicitly so
a future reader isn't surprised by it.

## Deliberately not done in Module 23

- **A real model runtime behind the API** — `/chat` is `FakeRuntime`-backed, this repo's
  standing honest-skip default since Module 6; `AppContext` takes the runtime via dependency
  injection, so swapping in `MLXRuntime`/`OllamaRuntime` on the resourced Mac needs no other
  change to the CLI or API.
- **`packages/local_ai_gateway/`** — reserved for Project 5 (Local inference gateway,
  curriculum.md §38), a later, differently-scoped unit of work (multi-client rate limiting,
  auth, routing). Not touched.
- **A local web UI or desktop wrapper** — curriculum's own deployment-modes table marks these as
  "best for demos/capstone" and "optional advanced packaging" respectively; both would call the
  same FastAPI service this module already builds, so building one now would exercise no new
  code.
- **A real Docker build** — this repo's own machine constraint (no model runtime installed) makes
  a Docker image with model support untestable here; documented as theory-only in the theory doc.
