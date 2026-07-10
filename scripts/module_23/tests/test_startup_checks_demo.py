import startup_checks_demo as sut


class TestRunLab:
    def test_healthy_configuration_passes_every_check(self):
        result = sut.run_lab()
        assert result["healthy_all_passed"] is True

    def test_broken_configuration_fails_the_catalog_check_only(self):
        result = sut.run_lab()
        assert result["broken_all_passed"] is False
        broken_by_name = {name: passed for name, passed, _ in result["broken_results"]}
        assert broken_by_name["model_catalog_parseable"] is False
        assert broken_by_name["config_valid"] is True
        assert broken_by_name["data_dir_writable"] is True

    def test_uses_a_real_temporary_directory_not_the_users_home(self):
        result = sut.run_lab()
        for _, _, detail in result["healthy_results"]:
            assert "/.local-llm-ai" not in detail


class TestResultToMarkdown:
    def test_markdown_reports_both_scenarios(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "Healthy configuration" in markdown
        assert "Broken configuration" in markdown
