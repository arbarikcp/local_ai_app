# Module 23 — Packaging and Deployment

> Phase: Production · Bible reference: [curriculum.md §33](../../curriculum.md#33-module-23--packaging-and-deployment)

## Goal

Package local AI apps for realistic use.

## This module builds the first composition root in the repo

Every prior module's lab scripts wired their own ad hoc subset of imports. Modules 1-22 built
real, tested pieces (runtimes, RAG, agents, fine-tuning, optimization, tracing, security) but
nothing ever assembled them into one running application. Module 23 is that assembly: a real
config loader, a real model registry (parsing Module 3's `MODEL_CATALOG.md` for the first time
programmatically), a real data directory layout, real startup/health/readiness checks, real
SQLite backup/restore, and `AppContext` — the first composition root, wiring runtime, gateway
admission control, security guard pipeline, and metrics together — then a real CLI (`typer`, a
long-declared but never-used dependency) and a real FastAPI service built on top of it.

> **Module boundary note:** `packages/local_ai_gateway/` (with its scaffolded `api/`, `auth/`,
> `rate_limits/`, `routing/`, `streaming/` subdirectories) is reserved for **Project 5 — Local
> inference gateway** (curriculum.md §38), a later unit of work with a different scope
> (multi-client rate limiting, auth, routing). This module does not touch it. Module 23's CLI and
> API service live under `scripts/module_23/` as runnable entry points, with their reusable logic
> in a new `packages/local_ai_core/deployment/` subpackage, matching every other module's
> package/script split.

## Reuse table

| Topic | Already implemented in |
|---|---|
| Concurrency control for the API service | `local_ai_core/gateway/admission_control.py`'s `AdmissionController` (Module 6.5) |
| Response/semantic caching | `local_ai_core/gateway/cache.py` (Module 6.5) |
| Security config (`allow_shell`, `allow_file_write`) | `local_ai_agents/policies/permissions.py`, `policies/approval.py` (Module 14) |
| `redact_pii_in_logs` | `local_ai_core/tracing/pii_redaction.py`, `structured_logging.py` (Module 21) |
| Request guarding | `local_ai_core/security/guard_pipeline.py`'s `RuleBasedGuardClassifier` (Module 22) |
| A model runtime | `local_ai_core/runtimes/fake.py`'s `FakeRuntime` (Module 6) — honest-skip default, same DI pattern every real-model adapter uses |

## New in this module

`deployment/config.py` (`AppConfig` — a real Pydantic model matching curriculum's exact YAML
config shape, `load_config()`), `deployment/model_registry.py` (`parse_model_catalog()` — the
first program to ever read `models/MODEL_CATALOG.md`'s embedded YAML fences), `deployment/data_dir.py`
(`ensure_data_dir_layout()` — real directory creation under a configurable `data_dir`),
`deployment/health.py` (`run_startup_checks()`, `run_readiness_check()` — real, executable
checks, not a checklist in prose), `deployment/backup.py` (`backup_sqlite_db()`/
`restore_sqlite_db()` — real `sqlite3` `.backup()` API, a genuine file-level round trip),
`deployment/app_context.py` (`AppContext`/`build_app_context()` — the composition root),
`scripts/module_23/cli_app.py` (a real `typer` CLI: `check`, `models`, `backup`, `restore`,
`serve`), `scripts/module_23/api_app.py` (a real FastAPI app: `/health`, `/ready`, `/models`,
`/chat` — guarded by Module 22's classifier, rate-bounded by Module 6.5's admission controller,
tested via `fastapi.testclient.TestClient`, no live server needed).

> **Machine note:** every piece here runs for real — config loading, model-registry parsing,
> data-directory creation, startup/health checks, SQLite backup/restore, the CLI, and the FastAPI
> service (via `TestClient`, no live socket needed) all execute with no model runtime installed.
> Only the `/chat` endpoint's actual generation is `FakeRuntime`-backed, the same honest-skip
> every module since Module 6 has used - swapping in `MLXRuntime` on the resourced Mac needs no
> other change, since `AppContext` takes the runtime via dependency injection.

## Core topics

### 1. Local CLI packaging

`scripts/module_23/cli_app.py` — a real `typer.Typer()` app (this repo's first actual use of
the `typer` dependency, declared since Phase 0 but never wired to anything). Commands: `check`
(runs startup checks), `models` (lists the parsed model registry), `backup`/`restore` (SQLite
backup/restore), `serve` (prints the uvicorn command to run the API service - doesn't block the
CLI process itself, since a lab script must return control).

### 2. Local API service

`scripts/module_23/api_app.py` — a real `FastAPI()` app: `/health` (liveness), `/ready`
(readiness - data dir + registry loaded), `/models` (registry contents), `/chat` (guarded by
`RuleBasedGuardClassifier`, admitted through `AdmissionController`, answered by the injected
runtime). Tested via `TestClient`, which drives the real ASGI app in-process - no live socket,
no honest-skip needed for the HTTP layer itself.

### 3. Local desktop-style service

Theory only: curriculum's deployment-modes table (below) - a desktop wrapper (e.g. a menu-bar
app shelling out to `uvicorn`) is a thin OS-integration layer over the same FastAPI service Lab
2 already builds; no new server logic, so no separate code.

### 4-5. Model download scripts, model registry

`model_registry.py`'s `parse_model_catalog()` - real parsing of `models/MODEL_CATALOG.md`'s
embedded YAML fences into `ModelRegistryEntry` objects. Model *download* stays theory/manual
(curriculum's own model-selection labs, Module 2/3's territory) - Module 23's registry answers
"what models does this app know about," not "fetch me a model."

### 6. Versioning

Theory, tied to this repo's own convention: every module ships as `git tag module-N` (this
module's own workflow, restated as the versioning strategy curriculum asks for - the app's
version *is* the tag of the commit it was built from).

### 7. Config management

`config.py`'s `AppConfig` - real Pydantic validation of curriculum's exact config shape
(`app`, `models`, `limits`, `security` sections), loaded from `config/app.example.yaml` (real,
committed).

### 8. Offline mode

Theory + a real config field: `AppConfig.app.offline_mode` (new field, not in curriculum's
literal example but implied by the topic) - when true, `AppContext` refuses to construct any
adapter whose `_real_*` functions would attempt a network call (documented, checked by a test
that the default `FakeRuntime` never touches the network regardless).

### 9-10. Startup checks, health checks

`health.py`'s `run_startup_checks()` (data directory writable, config valid, model catalog
parseable, sufficient free disk space - curriculum's own "disk pressure" concern from Module 20,
made checkable) vs. `run_readiness_check()` (fast, repeatable, what `/ready` calls on every
request) - two different real functions for two different curriculum topics, not one function
doing both jobs.

### 11. Data directory layout

`data_dir.py`'s `ensure_data_dir_layout()` - real subdirectories under `app.data_dir`
(`sessions/`, `audit/`, `adapters/`, `eval_feedback/`, `backups/`), one per this repo's existing
SQLite store (Modules 8.5, 14, 19, 21) - the first time any of those stores gets a real,
consistent on-disk home instead of a bespoke path per demo script.

### 12. Backup and restore

`backup.py`'s `backup_sqlite_db()`/`restore_sqlite_db()` - real `sqlite3` `.backup()` API
(the correct way to back up a live SQLite database, not a naive file copy that could catch a
half-written page), proven across an actual backup-then-restore-then-read round trip.

### 13. Runbooks

`docs/runbooks/operations_runbook.md` - a real, concrete runbook: starting the CLI/API,
running startup checks, backing up/restoring data, offline-mode notes, and a troubleshooting
table mapping this repo's own real error types (`RuntimeUnavailable`, `RequestTimeout`,
`SafetyPolicyViolation`, `QueueFullError`) to operator actions.

## Deployment modes

| Mode | Description | This module's artifact |
|---|---|---|
| CLI | best for developer tools and labs | `scripts/module_23/cli_app.py` |
| FastAPI local service | best for backend architecture | `scripts/module_23/api_app.py` |
| Local web UI | best for demos and capstone | out of scope - a later project's UI would call the same FastAPI service |
| Desktop wrapper | optional advanced packaging | theory only (§3 above) |
| Docker | useful for app dependencies but not always ideal for Mac GPU acceleration | theory only - this repo's own machine constraint (no model runtime installed) makes a Docker build untestable here anyway |

## Config example

```yaml
app:
  data_dir: ~/.local-llm-ai
  log_level: INFO

models:
  default_chat: llama3.2:3b
  default_extraction: gemma3:4b
  default_code: qwen2.5-coder:7b
  default_embedding: nomic-embed-text

limits:
  max_prompt_tokens: 6000
  max_output_tokens: 1024
  request_timeout_seconds: 60
  max_concurrent_requests: 1  # deliberate Mac-local default; see Module 6.5 concurrency benchmarks

security:
  allow_shell: false
  allow_file_write: approval_required
  redact_pii_in_logs: true
```

`config/app.example.yaml` (committed) is exactly this, loaded and validated for real by
`config.py`'s `load_config()`.

## Hands-on labs

1. **Package CLI** — `scripts/module_23/cli_app.py`.
2. **Package local API** — `scripts/module_23/api_app.py`.
3. **Add config file** — `config/app.example.yaml`, `deployment/config.py`.
4. **Add model registry** — `deployment/model_registry.py`.
5. **Add startup checks** — `deployment/health.py`.
6. **Add runbook** — `docs/runbooks/operations_runbook.md`.

## Deliverable

```text
config/
  app.example.yaml
packages/local_ai_core/deployment/
  config.py
  model_registry.py
  data_dir.py
  health.py
  backup.py
  app_context.py
  tests/
scripts/module_23/
  cli_app.py
  api_app.py
  config_and_registry_demo.py
  startup_checks_demo.py
docs/runbooks/
  operations_runbook.md
reports/module_23_packaging_report.md
```
