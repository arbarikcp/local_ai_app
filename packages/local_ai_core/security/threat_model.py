"""Threat modeling as real, importable data (theory doc §1, "Named risk
framework") — curriculum's attacker-controlled-surface list and OWASP LLM
Top 10 mapping table as real structures other code can reference, not
documentation prose a reader has to trust was kept in sync with the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThreatSurface(Enum):
    """Curriculum's exact "Attackers may control" list (theory doc §1)."""

    USER_PROMPT = "user_prompt"
    UPLOADED_DOCUMENT = "uploaded_document"
    WEB_PAGE = "web_page"
    FILENAME = "filename"
    METADATA = "metadata"
    TOOL_OUTPUT = "tool_output"
    CODE_COMMENT = "code_comment"
    DEPENDENCY_FILE = "dependency_file"
    TEST_DATA = "test_data"


@dataclass(frozen=True)
class OwaspRiskMapping:
    risk_area: str
    controls: list[str]


OWASP_RISK_MAP: list[OwaspRiskMapping] = [
    OwaspRiskMapping(
        risk_area="Prompt injection",
        controls=[
            "evals/prompt_injection.py::detect_prompt_injection_patterns",
            "security/guard_pipeline.py::RuleBasedGuardClassifier",
        ],
    ),
    OwaspRiskMapping(
        risk_area="Sensitive information disclosure",
        controls=[
            "tracing/pii_redaction.py::redact_pii",
            "tracing/structured_logging.py::PromptLoggingPolicy",
        ],
    ),
    OwaspRiskMapping(
        risk_area="Supply chain risk",
        controls=["security/supply_chain.py::verify_against_manifest"],
    ),
    OwaspRiskMapping(
        risk_area="Data and model poisoning",
        controls=["security/rag_ingestion_guard.py::screen_document_for_ingestion"],
    ),
    OwaspRiskMapping(
        risk_area="Improper output handling",
        controls=["security/guard_pipeline.py::enforce_guard_decision"],
    ),
    OwaspRiskMapping(
        risk_area="Excessive agency",
        controls=[
            "local_ai_agents/policies/budgets.py::ToolBudget",
            "local_ai_agents/planners/safety_budget.py::AgentSafetyBudget",
            "local_ai_agents/policies/approval.py::ApprovalGate",
        ],
    ),
    OwaspRiskMapping(
        risk_area="Insecure tool/plugin design",
        controls=[
            "local_ai_agents/executors/tool_executor.py::ToolExecutor",
            "local_ai_agents/tools/sandbox.py::resolve_within_sandbox",
            "security/tool_call_timeout.py::with_timeout",
        ],
    ),
]


def controls_for_risk(risk_area: str) -> list[str]:
    """Case-insensitive lookup - returns [] for an unmapped risk area rather
    than raising, since "no controls found" is meaningfully different from
    "this risk area doesn't exist" and callers may want to distinguish
    coverage gaps from typos separately.
    """
    for mapping in OWASP_RISK_MAP:
        if mapping.risk_area.lower() == risk_area.lower():
            return mapping.controls
    return []
