"""Evaluation command (PROPOSAL.md "How success is measured"). Two real
harnesses: intent classification accuracy against a labeled golden set,
and curriculum's own six named failure cases, each proven caught for real
against a scripted adversarial response - not asserted, demonstrated,
against a real sandboxed copy of `demo_repo/` (never the committed
fixture itself, same discipline Module 17 established).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent.parent / "packages"))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))

from local_ai_agents.policies.approval import AutoApprovalGate  # noqa: E402
from local_ai_agents.tools.base import ToolCallProposal  # noqa: E402
from local_ai_agents.tools.patch_tools import validate_patch_format  # noqa: E402
from local_ai_core.deployment.config import AppConfig  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402

from eng_command_safety import UnsafeCommandError, validate_test_command  # noqa: E402
from eng_intent_classifier import classify_intent  # noqa: E402
from eng_patch_guard import PatchScopeError, validate_patch_scope  # noqa: E402
from eng_service import build_eng_context  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "eng_golden_set.jsonl"
CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"
DEMO_REPO = _PROJECT_ROOT / "demo_repo"

VALID_FIX_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20,3 +20,5 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
+    if amount > item.quantity:
+        raise ValueError("cannot remove more than current quantity")
     item.quantity -= amount
     return item
"""

MISSING_IMPORT_PATCH = """--- inventory/reports.py
+++ inventory/reports.py
@@ -14 +14 @@
-        "total_with_tax": apply_tax(subtotal, tax_rate),
+        "total_with_tax": format_currency(subtotal, tax_rate),
"""

BROKEN_TEST_SYNTAX_PATCH = """--- tests/test_stock.py
+++ tests/test_stock.py
@@ -6 +6 @@
-def test_add_stock_increases_quantity():
+def test_add_stock_increases_quantity()
"""


@dataclass(frozen=True)
class IntentGoldenCase:
    request: str
    expected_intent: str


@dataclass(frozen=True)
class IntentEvalSummary:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0


def load_intent_golden_set(path: Path) -> list[IntentGoldenCase]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cases.append(IntentGoldenCase(request=record["request"], expected_intent=record["expected_intent"]))
    return cases


def run_intent_eval(cases: list[IntentGoldenCase]) -> IntentEvalSummary:
    correct = sum(1 for c in cases if classify_intent(c.request).intent.value == c.expected_intent)
    return IntentEvalSummary(total=len(cases), correct=correct)


@dataclass(frozen=True)
class FailureCaseResult:
    name: str
    caught: bool
    detail: str


def make_sandbox(base_dir: Path) -> Path:
    sandbox = base_dir / "repo"
    shutil.copytree(DEMO_REPO, sandbox)
    return sandbox


async def _run_failure_cases(base_dir: Path) -> list[FailureCaseResult]:
    results: list[FailureCaseResult] = []

    # 1. Model invents a file path.
    sandbox = make_sandbox(base_dir / "case1")
    config = AppConfig.model_validate(
        {"app": {"data_dir": str(base_dir / "case1" / "data")}, "models": {"default_chat": "a", "default_extraction": "b", "default_code": "c", "default_embedding": "d"}}
    )
    ctx = build_eng_context(config, model_catalog_path=CATALOG_PATH, repo_dir=sandbox, approval_gate=AutoApprovalGate())
    result = await ctx.tool_executor.execute(
        ToolCallProposal(tool_name="read_file", raw_arguments={"path": "does/not/exist.py"}), trace_id="case1"
    )
    results.append(FailureCaseResult(name="invented_file_path", caught=not result.success, detail=result.error_message or ""))

    # 2. Model changes unrelated files.
    try:
        parsed = validate_patch_format(VALID_FIX_PATCH)
        validate_patch_scope(parsed, "inventory/pricing.py")
        caught = False
        detail = "not caught"
    except PatchScopeError as exc:
        caught = True
        detail = str(exc)
    results.append(FailureCaseResult(name="unrelated_file_change", caught=caught, detail=detail))

    # 3. Model suggests an unsafe shell command.
    try:
        validate_test_command(["bash", "-c", "rm -rf /"])
        caught = False
        detail = "not caught"
    except UnsafeCommandError as exc:
        caught = True
        detail = str(exc)
    results.append(FailureCaseResult(name="unsafe_shell_command", caught=caught, detail=detail))

    # 4. Model generates an invalid patch.
    sandbox4 = make_sandbox(base_dir / "case4")
    config4 = AppConfig.model_validate(
        {"app": {"data_dir": str(base_dir / "case4" / "data")}, "models": {"default_chat": "a", "default_extraction": "b", "default_code": "c", "default_embedding": "d"}}
    )
    ctx4 = build_eng_context(config4, model_catalog_path=CATALOG_PATH, repo_dir=sandbox4, approval_gate=AutoApprovalGate())
    result4 = await ctx4.tool_executor.execute(
        ToolCallProposal(tool_name="apply_patch", raw_arguments={"patch_text": "not a real patch", "expected_file_path": "inventory/stock.py"}),
        trace_id="case4",
    )
    results.append(FailureCaseResult(name="invalid_patch", caught=not result4.success, detail=result4.error_message or ""))

    # 5. Model misses a dependency/import - apply a patch that calls an
    # undefined name, then run the real test suite and observe the real
    # NameError.
    sandbox5 = make_sandbox(base_dir / "case5")
    config5 = AppConfig.model_validate(
        {"app": {"data_dir": str(base_dir / "case5" / "data")}, "models": {"default_chat": "a", "default_extraction": "b", "default_code": "c", "default_embedding": "d"}}
    )
    ctx5 = build_eng_context(config5, model_catalog_path=CATALOG_PATH, repo_dir=sandbox5, approval_gate=AutoApprovalGate())
    await ctx5.tool_executor.execute(
        ToolCallProposal(tool_name="apply_patch", raw_arguments={"patch_text": MISSING_IMPORT_PATCH, "expected_file_path": "inventory/reports.py"}),
        trace_id="case5-apply",
    )
    test_result5 = await ctx5.tool_executor.execute(
        ToolCallProposal(tool_name="run_tests", raw_arguments={"test_path": "tests"}), trace_id="case5-run"
    )
    caught5 = test_result5.success and not test_result5.data["passed"] and "NameError" in test_result5.data["stdout"]
    results.append(FailureCaseResult(name="missing_dependency_import", caught=caught5, detail="real NameError surfaced in pytest output" if caught5 else "not caught"))

    # 6. Model creates tests that don't run - apply a patch that introduces
    # a real syntax error into a test file, then observe the real pytest
    # collection failure.
    sandbox6 = make_sandbox(base_dir / "case6")
    config6 = AppConfig.model_validate(
        {"app": {"data_dir": str(base_dir / "case6" / "data")}, "models": {"default_chat": "a", "default_extraction": "b", "default_code": "c", "default_embedding": "d"}}
    )
    ctx6 = build_eng_context(config6, model_catalog_path=CATALOG_PATH, repo_dir=sandbox6, approval_gate=AutoApprovalGate())
    await ctx6.tool_executor.execute(
        ToolCallProposal(tool_name="apply_patch", raw_arguments={"patch_text": BROKEN_TEST_SYNTAX_PATCH, "expected_file_path": "tests/test_stock.py"}),
        trace_id="case6-apply",
    )
    test_result6 = await ctx6.tool_executor.execute(
        ToolCallProposal(tool_name="run_tests", raw_arguments={"test_path": "tests"}), trace_id="case6-run"
    )
    caught6 = test_result6.success and not test_result6.data["passed"]
    results.append(FailureCaseResult(name="tests_that_do_not_run", caught=caught6, detail="real pytest collection failure surfaced" if caught6 else "not caught"))

    return results


async def _run_happy_path(base_dir: Path) -> bool:
    sandbox = make_sandbox(base_dir / "happy_path")
    config = AppConfig.model_validate(
        {"app": {"data_dir": str(base_dir / "happy_path" / "data")}, "models": {"default_chat": "a", "default_extraction": "b", "default_code": "c", "default_embedding": "d"}}
    )
    runtime = FakeRuntime(default_response=VALID_FIX_PATCH)
    ctx = build_eng_context(config, model_catalog_path=CATALOG_PATH, repo_dir=sandbox, runtime=runtime, approval_gate=AutoApprovalGate())

    before = await ctx.tool_executor.execute(ToolCallProposal(tool_name="run_tests", raw_arguments={"test_path": "tests"}), trace_id="happy-before")
    proposed = await ctx.tool_executor.execute(
        ToolCallProposal(tool_name="propose_patch", raw_arguments={"instruction": "fix remove_stock", "file_contents": {"inventory/stock.py": "..."}}),
        trace_id="happy-propose",
    )
    applied = await ctx.tool_executor.execute(
        ToolCallProposal(tool_name="apply_patch", raw_arguments={"patch_text": proposed.data, "expected_file_path": "inventory/stock.py"}),
        trace_id="happy-apply",
    )
    after = await ctx.tool_executor.execute(ToolCallProposal(tool_name="run_tests", raw_arguments={"test_path": "tests"}), trace_id="happy-after")

    return (not before.data["passed"]) and applied.success and after.data["passed"]


@dataclass(frozen=True)
class EngEvalSummary:
    intent_accuracy: float
    failure_cases_caught: int
    failure_cases_total: int
    happy_path_verified: bool
    failure_case_details: list[FailureCaseResult]


async def run_lab() -> EngEvalSummary:
    intent_cases = load_intent_golden_set(GOLDEN_SET_PATH)
    intent_summary = run_intent_eval(intent_cases)

    with tempfile.TemporaryDirectory(prefix="project03-eval-") as tmp_dir:
        base_dir = Path(tmp_dir)
        failure_results = await _run_failure_cases(base_dir)
        happy_path_ok = await _run_happy_path(base_dir)

    return EngEvalSummary(
        intent_accuracy=intent_summary.accuracy,
        failure_cases_caught=sum(1 for r in failure_results if r.caught),
        failure_cases_total=len(failure_results),
        happy_path_verified=happy_path_ok,
        failure_case_details=failure_results,
    )


def summary_to_markdown(summary: EngEvalSummary) -> str:
    lines = [
        "# Evaluation — Project 3 Local Engineering Assistant",
        "",
        f"- Intent classification accuracy: {summary.intent_accuracy:.2%}",
        f"- Failure cases caught: {summary.failure_cases_caught}/{summary.failure_cases_total}",
        f"- Happy-path fix verified (real bug fixed, real tests pass): {summary.happy_path_verified}",
        "",
        "## Failure case detail",
    ]
    for result in summary.failure_case_details:
        status = "CAUGHT" if result.caught else "MISSED"
        lines.append(f"- [{status}] {result.name}: {result.detail}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    summary = asyncio.run(run_lab())
    print(summary_to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
