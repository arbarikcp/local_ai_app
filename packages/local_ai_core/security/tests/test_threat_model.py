from local_ai_core.security.threat_model import OWASP_RISK_MAP, ThreatSurface, controls_for_risk


class TestThreatSurface:
    def test_has_every_curriculum_surface(self):
        expected = {
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
        assert {s.value for s in ThreatSurface} == expected


class TestOwaspRiskMap:
    def test_has_all_seven_curriculum_risk_areas(self):
        expected = {
            "Prompt injection",
            "Sensitive information disclosure",
            "Supply chain risk",
            "Data and model poisoning",
            "Improper output handling",
            "Excessive agency",
            "Insecure tool/plugin design",
        }
        assert {m.risk_area for m in OWASP_RISK_MAP} == expected

    def test_every_risk_area_has_at_least_one_control(self):
        for mapping in OWASP_RISK_MAP:
            assert len(mapping.controls) > 0


class TestControlsForRisk:
    def test_returns_the_controls_for_a_known_risk_area(self):
        controls = controls_for_risk("Prompt injection")
        assert any("guard_pipeline" in c for c in controls)

    def test_lookup_is_case_insensitive(self):
        assert controls_for_risk("prompt injection") == controls_for_risk("Prompt injection")

    def test_unmapped_risk_area_returns_empty_list(self):
        assert controls_for_risk("made up risk area") == []
