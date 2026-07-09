"""RuleBasedGuardClassifier — the real, default implementation of
curriculum's guard-model pipeline (theory doc "Local guard models", §2-5).
This machine runs no model at all, so this classifier is deterministic
rule-based composition of three already-real detectors (prompt injection,
PII, secrets) rather than an ML classifier - it satisfies the same
`GuardClassifier` Protocol a future Llama-Guard-style model-backed
classifier could implement on the resourced Mac, the same DI shape Module
6's `MLXRuntime` established. "A guard model is a signal, not an
authority" (curriculum's own words): `enforce_guard_decision()` is the
policy layer that actually acts on the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from local_ai_core.evals.prompt_injection import detect_prompt_injection_patterns
from local_ai_core.runtimes.errors import SafetyPolicyViolation
from local_ai_core.security.secrets_scanner import scan_for_secrets
from local_ai_core.tracing.pii_redaction import redact_pii


class GuardVerdict(Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


@dataclass(frozen=True)
class GuardSignal:
    category: str
    detail: str


@dataclass(frozen=True)
class GuardDecision:
    verdict: GuardVerdict
    signals: list[GuardSignal] = field(default_factory=list)
    reason: str = ""


class GuardClassifier(Protocol):
    def classify(self, text: str) -> GuardDecision: ...


class RuleBasedGuardClassifier:
    """BLOCKs on any detected prompt-injection pattern (the highest-severity
    signal - injected content can hijack downstream tool calls, curriculum's
    own "excessive agency" risk). FLAGs on PII or secrets presence alone
    (needs redaction/handling before logging or forwarding, but isn't by
    itself evidence of an attack). ALLOWs otherwise.
    """

    def classify(self, text: str) -> GuardDecision:
        signals: list[GuardSignal] = []

        injection_patterns = detect_prompt_injection_patterns(text)
        for pattern in injection_patterns:
            signals.append(GuardSignal(category="prompt_injection", detail=pattern))

        redaction = redact_pii(text)
        for category, count in redaction.redaction_counts.items():
            signals.append(GuardSignal(category="pii", detail=f"{category} x{count}"))

        secrets = scan_for_secrets(text)
        for category, count in secrets.matches.items():
            signals.append(GuardSignal(category="secrets", detail=f"{category} x{count}"))

        if injection_patterns:
            return GuardDecision(
                verdict=GuardVerdict.BLOCK,
                signals=signals,
                reason=f"{len(injection_patterns)} prompt-injection pattern(s) matched",
            )
        if redaction.total_redactions or secrets.found_secrets:
            return GuardDecision(
                verdict=GuardVerdict.FLAG,
                signals=signals,
                reason="PII and/or secrets detected - requires redaction before logging/forwarding",
            )
        return GuardDecision(verdict=GuardVerdict.ALLOW, signals=signals, reason="no signals detected")


def enforce_guard_decision(decision: GuardDecision) -> None:
    """The first real caller of `runtimes/errors.py`'s `SafetyPolicyViolation`
    - declared since Module 6, never previously raised anywhere in this
    repo. A BLOCK verdict is a policy violation by definition; FLAG/ALLOW
    are not - a caller that wants to act on FLAG (e.g. redact before
    logging) inspects `decision.signals` itself rather than this function
    raising for a case that isn't actually a violation.
    """
    if decision.verdict == GuardVerdict.BLOCK:
        raise SafetyPolicyViolation(decision.reason)
