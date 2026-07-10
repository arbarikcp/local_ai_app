import shutil
from pathlib import Path

from typer.testing import CliRunner

import eng_cli as sut

runner = CliRunner()
DEMO_REPO = Path(__file__).resolve().parent.parent / "demo_repo"

CONFIG_YAML = """
app:
  data_dir: {data_dir}
models:
  default_chat: llama3.2:3b
  default_extraction: gemma3:4b
  default_code: qwen2.5-coder:7b
  default_embedding: nomic-embed-text
"""

VALID_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20,3 +20,5 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
+    if amount > item.quantity:
+        raise ValueError("cannot remove more than current quantity")
     item.quantity -= amount
     return item
"""


def make_sandbox(tmp_path) -> Path:
    sandbox = tmp_path / "repo"
    shutil.copytree(DEMO_REPO, sandbox)
    return sandbox


def write_config(tmp_path) -> str:
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(CONFIG_YAML.format(data_dir=str(tmp_path / "data")))
    return str(config_path)


class TestExplainRepo:
    def test_lists_real_symbols_across_the_sandboxed_repo(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        result = runner.invoke(sut.app, ["explain-repo", "--repo-dir", str(sandbox), "--config-path", config_path])
        assert result.exit_code == 0
        assert "remove_stock" in result.output
        assert "calculate_discount" in result.output


class TestSearch:
    def test_finds_real_matches(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        result = runner.invoke(sut.app, ["search", "quantity", "--repo-dir", str(sandbox), "--config-path", config_path])
        assert result.exit_code == 0
        assert "stock.py" in result.output


class TestExplainSymbol:
    def test_a_known_symbol_gets_a_real_fake_runtime_explanation(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        result = runner.invoke(
            sut.app, ["explain-symbol", "remove_stock", "--repo-dir", str(sandbox), "--config-path", config_path]
        )
        assert result.exit_code == 0

    def test_an_unknown_symbol_exits_nonzero(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        result = runner.invoke(
            sut.app, ["explain-symbol", "does_not_exist", "--repo-dir", str(sandbox), "--config-path", config_path]
        )
        assert result.exit_code != 0


class TestApplyPatchRequiresApproval:
    def test_without_approve_flag_the_patch_is_denied(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        patch_file = tmp_path / "fix.patch"
        patch_file.write_text(VALID_PATCH)

        result = runner.invoke(
            sut.app,
            ["apply-patch", str(patch_file), "inventory/stock.py", "--repo-dir", str(sandbox), "--config-path", config_path],
        )
        assert result.exit_code != 0
        assert "cannot remove more than current quantity" not in (sandbox / "inventory" / "stock.py").read_text()

    def test_with_approve_flag_the_patch_is_applied(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        patch_file = tmp_path / "fix.patch"
        patch_file.write_text(VALID_PATCH)

        result = runner.invoke(
            sut.app,
            [
                "apply-patch", str(patch_file), "inventory/stock.py",
                "--repo-dir", str(sandbox), "--config-path", config_path, "--approve",
            ],
        )
        assert result.exit_code == 0
        assert "cannot remove more than current quantity" in (sandbox / "inventory" / "stock.py").read_text()


class TestRunTestsRequiresApproval:
    def test_without_approve_flag_tests_do_not_run(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        result = runner.invoke(sut.app, ["run-tests", "--repo-dir", str(sandbox), "--config-path", config_path])
        assert result.exit_code != 0

    def test_with_approve_flag_tests_run_and_report_the_real_failure(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        result = runner.invoke(sut.app, ["run-tests", "--repo-dir", str(sandbox), "--config-path", config_path, "--approve"])
        assert result.exit_code != 0  # the real bug is still present
        assert "failed" in result.output.lower()

    def test_after_a_real_fix_tests_pass(self, tmp_path):
        sandbox = make_sandbox(tmp_path)
        config_path = write_config(tmp_path)
        patch_file = tmp_path / "fix.patch"
        patch_file.write_text(VALID_PATCH)
        runner.invoke(
            sut.app,
            [
                "apply-patch", str(patch_file), "inventory/stock.py",
                "--repo-dir", str(sandbox), "--config-path", config_path, "--approve",
            ],
        )

        result = runner.invoke(sut.app, ["run-tests", "--repo-dir", str(sandbox), "--config-path", config_path, "--approve"])
        assert result.exit_code == 0
