import red_team_dataset_demo as sut


class TestRunLab:
    def test_matches_the_real_committed_dataset_size(self):
        result = sut.run_lab()
        assert result["total"] == 39
        assert result["malicious_count"] == 25
        assert result["benign_count"] == 14

    def test_every_curriculum_threat_surface_is_represented(self):
        result = sut.run_lab()
        expected_surfaces = {
            "user_prompt",
            "uploaded_document",
            "web_page",
            "filename",
            "metadata",
            "tool_output",
            "code_comment",
            "dependency_file",
            "test_data",
        }
        assert set(result["surface_counts"].keys()) == expected_surfaces


class TestResultToMarkdown:
    def test_markdown_reports_the_total(self):
        result = sut.run_lab()
        markdown = sut.result_to_markdown(result)
        assert "39" in markdown
