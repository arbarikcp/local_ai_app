# Module 22 — Security, Privacy, and Red Teaming

> Phase: Production · Bible reference: [curriculum.md §32](../../curriculum.md#32-module-22--security-privacy-and-red-teaming)

## Goal

Secure local AI applications against realistic threats.

## This module composes four modules' worth of real controls, then fills the gaps

Modules 14-16 already built and shipped real, tested security infrastructure — permission
allowlists, an approval workflow, an audit log, tool budgets, loop prevention, path sandboxing,
and a first-pass prompt-injection pattern screen, all wired into a real MCP-like server (Module
16's `mcp_like_server.py`). Module 21 built real PII redaction and policy-driven prompt logging.
Module 22's job is threefold: (1) name that reuse explicitly against curriculum's 14 topics and
OWASP mapping, (2) implement the pieces that were declared but never built (`SafetyPolicyViolation`
has sat unused in `runtimes/errors.py` since Module 6), and (3) build what's genuinely missing —
a guard-classifier pipeline, RAG ingestion screening, model supply-chain verification, secrets
detection, and per-tool-call timeouts.

> **Module boundary note:** this module only adds new code under
> `packages/local_ai_core/security/` — it imports from `local_ai_agents`, `local_ai_core.evals`,
> and `local_ai_core.tracing` but never edits their files, per this repo's "stay within your
> module's folder boundary" rule. Two independent regex-based injection screens already exist
> (`evals/prompt_injection.py`, wired into Module 16's server, and `prompts/injection_guard.py`,
> unwired/dead code) — Module 22 treats `evals/prompt_injection.py` as canonical (it's the one
> actually protecting a real code path) and does not modify either file; the duplication is
> flagged in this module's report as a finding, not silently fixed across a module boundary.

## Reuse table

| Topic | Already implemented in |
|---|---|
| Insecure tool design (allowlists, schemas, approval, audit) | `local_ai_agents/policies/permissions.py`, `executors/tool_executor.py` (Pydantic validation), `policies/approval.py`, `policies/audit_log.py` |
| Human approval | `local_ai_agents/policies/approval.py`'s `ApprovalGate`, wired into `tool_executor.py` and `executors/workflow_executor.py` |
| Local file access / sandboxing | `local_ai_agents/tools/sandbox.py`'s `resolve_within_sandbox()` |
| Excessive agency (budgets, loop prevention) | `local_ai_agents/policies/budgets.py`'s `ToolBudget`, `planners/safety_budget.py`'s `AgentSafetyBudget`, `planners/loop_prevention.py`'s `LoopGuard` |
| Prompt injection (direct + indirect, first pass) | `local_ai_core/evals/prompt_injection.py`'s `detect_prompt_injection_patterns()`, wired into Module 16's `mcp_like_server.py` for both tool descriptions and resource content |
| Sensitive information disclosure | `local_ai_core/tracing/pii_redaction.py`'s `redact_pii()` |
| Logging privacy | `local_ai_core/tracing/structured_logging.py`'s `PromptLoggingPolicy` |
| DoS (global concurrency) | `local_ai_core/gateway/admission_control.py`, `gateway/queue.py`'s `BoundedRequestQueue` |

## New in this module

`security/threat_model.py` (real `ThreatSurface` enum + `OWASP_RISK_MAP`, a real importable
data structure, not documentation prose), `security/guard_pipeline.py` (`RuleBasedGuardClassifier`
— composes the reused injection screen and PII/secrets detectors into one `GuardDecision`, and
`enforce_guard_decision()` — the first real caller of `runtimes/errors.py`'s long-declared-but-
dead `SafetyPolicyViolation`), `security/guard_eval.py` (real catch-rate/false-positive-rate/
latency measurement, Labs 6-7), `security/rag_ingestion_guard.py` (RAG data-poisoning screening
before ingestion), `security/supply_chain.py` (real SHA-256 checksum verification against a
manifest), `security/secrets_scanner.py` (real regex-based credential detection, a sibling to
Module 21's `pii_redaction.py`), `security/tool_call_timeout.py` (per-call timeout enforcement,
reusing `runtimes/errors.py`'s `RequestTimeout` rather than declaring a fourth timeout type).

> **Machine note:** curriculum's "local guard models" section explicitly wants real ML
> classifiers (Llama-Guard/Granite-Guardian-style) evaluated as security components. This
> machine runs no model at all, so `RuleBasedGuardClassifier` is the real, deterministic,
> testable default implementation — matching the `GuardClassifier` Protocol a future model-backed
> classifier could implement on the resourced Mac, the same DI shape Module 6's `MLXRuntime`
> established. "A guard model is a signal, not an authority" (curriculum's own words) — the
> deterministic policy layer (Module 14's `ToolExecutor`, `enforce_guard_decision()`) still
> decides what actually happens.

## Core topics

### 1. Threat modeling

`threat_model.py`'s `ThreatSurface` enum — curriculum's exact attacker-controlled-surface list
(user prompts, uploaded documents, web pages, filenames, metadata, tool outputs, code comments,
dependency files, test data) as a real enum other code can reference, not a bulleted list.

### 2-3. Prompt injection, indirect prompt injection

Reused — `evals/prompt_injection.py`, already screening both direct user input (via
`guard_pipeline.py`'s classifier) and indirect injection surfaces (tool descriptions, retrieved
resource content, per Module 16).

### 4. Sensitive data disclosure

Reused — `tracing/pii_redaction.py` (Module 21), composed into `guard_pipeline.py`'s classifier
as one of its three signal sources.

### 5. Insecure output handling

`guard_pipeline.py`'s `GuardVerdict.BLOCK` path, enforced via `enforce_guard_decision()` raising
a real `SafetyPolicyViolation` — output a guard classifies as carrying an injection payload
never reaches a caller un-blocked.

### 6. Insecure tool design

Reused — Module 14's full `ToolExecutor` pipeline (registry lookup, permission check, Pydantic
schema validation, approval gate, budget enforcement, audit log), extended here with
`tool_call_timeout.py`'s per-call timeout wrapper (the one gap: nothing previously bounded a
hanging tool handler).

### 7. RAG data poisoning

`rag_ingestion_guard.py`'s `screen_document_for_ingestion()` — every document is screened for
injection patterns before ingestion regardless of declared source trust (a compromised
"trusted" source is still a real threat curriculum's own threat model names), returning a real
`IngestionDecision` with the specific patterns that triggered quarantine.

### 8. Model supply chain

`supply_chain.py`'s `verify_against_manifest()` — real SHA-256 checksum computation and
comparison against a `ModelManifestEntry` (name, source URL, checksum, license), the same
content-hashing discipline Module 19's `hash_dataset()` and `local_ai_rag`'s incremental
indexer already established for datasets, applied here to model files.

### 9. Secrets handling

`secrets_scanner.py`'s `scan_for_secrets()` — real regex detection for AWS-shaped access keys,
generic API key assignments, PEM private-key headers, and bearer tokens. Distinct from PII: a
secret should never be logged even redacted-in-place (the redacted placeholder itself is safe;
a "sk-...7f2a" partial reveal is not), so this module treats any secret match as a hard signal
for `guard_pipeline.py`'s classifier, not a redact-and-continue case.

### 10. Logging privacy

Reused — Module 21's `PromptLoggingPolicy`, now with a real default `redactor` wiring
demonstrated in a lab script: `pii_redaction.redact_pii` composed directly into
`StructuredLogger.log_prompt()`'s injectable `redactor` parameter.

### 11. Local file access

Reused — Module 14's `sandbox.py`.

### 12. Sandboxing

Reused (path sandboxing) plus new (`tool_call_timeout.py`, execution-time sandboxing) — a tool
handler is bounded by both where it can touch (`sandbox.py`) and how long it may run
(`tool_call_timeout.py`).

### 13. Human approval

Reused — Module 14/15's `ApprovalGate`.

### 14. Red-team testing

`datasets/red_team/` — a real, committed, hand-labeled set of malicious prompts, documents, and
tool-call requests (continuing this course's Nimbus Cloud Storage theme), each labeled
`is_malicious: bool` for `guard_eval.py`'s catch-rate measurement.

## Named risk framework: OWASP LLM Top 10 mapping

`threat_model.OWASP_RISK_MAP` — curriculum's exact table, each entry citing the real
class/function that enforces it:

| OWASP-style risk area | Course control |
|---|---|
| Prompt injection | `evals/prompt_injection.py`, `guard_pipeline.py`'s `RuleBasedGuardClassifier` |
| Sensitive information disclosure | `tracing/pii_redaction.py`, `tracing/structured_logging.py`'s `PromptLoggingPolicy` |
| Supply chain risk | `security/supply_chain.py`'s `verify_against_manifest()` |
| Data and model poisoning | `security/rag_ingestion_guard.py`'s `screen_document_for_ingestion()` |
| Improper output handling | `guard_pipeline.py`'s `enforce_guard_decision()` raising `SafetyPolicyViolation` |
| Excessive agency | `local_ai_agents/policies/budgets.py`, `planners/safety_budget.py`, `policies/approval.py` |
| Insecure tool/plugin design | `local_ai_agents/executors/tool_executor.py`, `tools/sandbox.py`, `security/tool_call_timeout.py` |

## Local guard models

```text
user turn/document/tool result
  -> injection/safety classifier   (guard_pipeline.RuleBasedGuardClassifier)
  -> policy decision               (guard_pipeline.enforce_guard_decision -> SafetyPolicyViolation)
  -> allowed context to generator
  -> output safety check if needed (same classifier, reused on model output)
  -> audit log                     (local_ai_agents.policies.audit_log.AuditLog, reused)
```

## Hands-on labs

1. **Build red-team prompt dataset** — `datasets/red_team/`,
   `scripts/module_22/red_team_dataset_demo.py`.
2. **Attack RAG with malicious document** — `scripts/module_22/rag_poisoning_demo.py`.
3. **Attack tool calling with injected tool request** —
   `scripts/module_22/tool_injection_demo.py`.
4. **Add policy enforcement** — `guard_pipeline.py`'s `enforce_guard_decision()`, same script.
5. **Add approval workflow** — reuses Module 14/15's `ApprovalGate` unchanged, same script.
6. **Run local guard models/classifiers against the red-team set** —
   `scripts/module_22/guard_classifier_eval_demo.py`.
7. **Measure catch rate, false positives, false negatives, and latency** —
   `guard_eval.py`'s `evaluate_guard_classifier()`, same script.
8. **Produce security report mapped to OWASP LLM risks** —
   `reports/module_22_security_report.md`.

## Deliverable

```text
datasets/red_team/
  red_team_prompts.jsonl
packages/local_ai_core/security/
  threat_model.py
  guard_pipeline.py
  guard_eval.py
  rag_ingestion_guard.py
  supply_chain.py
  secrets_scanner.py
  tool_call_timeout.py
  tests/
scripts/module_22/
  red_team_dataset_demo.py
  rag_poisoning_demo.py
  tool_injection_demo.py
  guard_classifier_eval_demo.py
reports/module_22_security_report.md
```
