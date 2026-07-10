from typer.testing import CliRunner

import cli_app as sut

runner = CliRunner()

CONFIG_YAML = """
app:
  data_dir: {data_dir}
models:
  default_chat: llama3.2:3b
  default_extraction: gemma3:4b
  default_code: qwen2.5-coder:7b
  default_embedding: nomic-embed-text
"""


def write_test_config(tmp_path) -> str:
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(CONFIG_YAML.format(data_dir=str(tmp_path / "data")))
    return str(config_path)


class TestCheckCommand:
    def test_all_checks_pass_and_exits_zero(self, tmp_path):
        config_path = write_test_config(tmp_path)
        result = runner.invoke(sut.app, ["check", "--config-path", config_path])
        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "FAIL" not in result.output


class TestModelsCommand:
    def test_lists_the_real_catalog(self, tmp_path):
        config_path = write_test_config(tmp_path)
        result = runner.invoke(sut.app, ["models", "--config-path", config_path])
        assert result.exit_code == 0
        assert "qwen2.5:1.5b-instruct" in result.output
        assert result.output.count("(") == 10


class TestBackupAndRestoreCommands:
    def test_backup_then_list_then_restore_round_trip(self, tmp_path):
        config_path = write_test_config(tmp_path)

        backup_result = runner.invoke(sut.app, ["backup", "--config-path", config_path])
        assert backup_result.exit_code == 0
        assert "Backed up to" in backup_result.output

        list_result = runner.invoke(sut.app, ["list-backup-files", "--config-path", config_path])
        assert list_result.exit_code == 0
        backup_file = list_result.output.strip().splitlines()[-1]

        restore_result = runner.invoke(sut.app, ["restore", backup_file, "--config-path", config_path])
        assert restore_result.exit_code == 0
        assert "Restored" in restore_result.output


class TestServeCommand:
    def test_prints_the_uvicorn_command_without_blocking(self, tmp_path):
        config_path = write_test_config(tmp_path)
        result = runner.invoke(sut.app, ["serve", "--config-path", config_path])
        assert result.exit_code == 0
        assert "uvicorn" in result.output
