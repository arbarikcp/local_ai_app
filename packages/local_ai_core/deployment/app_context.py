"""AppContext — the first composition root in this repo. Every prior
module's demo script wired its own ad hoc subset of imports; this wires
runtime, gateway admission control, the security guard pipeline, and
metrics into one real, DI-friendly object the CLI and API service both
build from (theory doc "This module builds the first composition root").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.config import AppConfig
from local_ai_core.deployment.data_dir import DataDirectoryLayout, ensure_data_dir_layout
from local_ai_core.deployment.model_registry import ModelRegistry, load_model_registry
from local_ai_core.gateway.admission_control import AdmissionController, AdmissionPolicy
from local_ai_core.optimization.dashboard import InMemoryMetricsHook
from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.security.guard_pipeline import GuardClassifier, RuleBasedGuardClassifier

from local_ai_agents.policies.audit_log import AuditLog


@dataclass
class AppContext:
    config: AppConfig
    data_dir: DataDirectoryLayout
    model_registry: ModelRegistry
    runtime: LLMRuntime
    admission_controller: AdmissionController
    guard_classifier: GuardClassifier
    audit_log: AuditLog
    metrics_hook: InMemoryMetricsHook


def build_app_context(
    config: AppConfig,
    *,
    model_catalog_path: str | Path,
    runtime: LLMRuntime | None = None,
) -> AppContext:
    """Wires everything for real except the model runtime itself, which
    defaults to `FakeRuntime` - this repo's standing honest-skip default
    (Module 6 onward). Inject a real `MLXRuntime`/`OllamaRuntime` on the
    resourced Mac; nothing else in `AppContext` changes.
    """
    data_dir = ensure_data_dir_layout(config)
    model_registry = load_model_registry(model_catalog_path)
    metrics_hook = InMemoryMetricsHook()
    resolved_runtime = runtime or FakeRuntime(
        default_response="(FakeRuntime - no model runtime installed on this machine)",
        metrics_hook=metrics_hook,
    )

    admission_controller = AdmissionController(
        AdmissionPolicy(
            max_concurrent_requests=config.limits.max_concurrent_requests,
            reason=f"from config: limits.max_concurrent_requests={config.limits.max_concurrent_requests}",
        )
    )
    guard_classifier = RuleBasedGuardClassifier()
    audit_log = AuditLog(data_dir.audit_db)

    return AppContext(
        config=config,
        data_dir=data_dir,
        model_registry=model_registry,
        runtime=resolved_runtime,
        admission_controller=admission_controller,
        guard_classifier=guard_classifier,
        audit_log=audit_log,
        metrics_hook=metrics_hook,
    )
