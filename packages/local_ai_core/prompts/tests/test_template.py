from local_ai_core.prompts.few_shot import FewShotExample, NegativeExample
from local_ai_core.prompts.template import PromptTemplate


def _minimal_template(**overrides) -> PromptTemplate:
    defaults = dict(
        prompt_id="extraction",
        version="v1",
        role="You are an information extraction engine.",
        task="Extract the requested fields from the input text.",
    )
    defaults.update(overrides)
    return PromptTemplate(**defaults)


class TestRenderSectionOrder:
    def test_sections_appear_in_the_canonical_order(self):
        template = _minimal_template(
            input_contract="Plain text.",
            output_contract="Strict JSON.",
            rules=["Return only valid JSON.", "Use null for missing fields."],
        )
        rendered = template.render("Maria is 29.")
        role_idx = rendered.index("You are an information extraction engine.")
        task_idx = rendered.index("Task:")
        input_contract_idx = rendered.index("Input contract:")
        output_contract_idx = rendered.index("Output contract:")
        rules_idx = rendered.index("Rules:")
        input_idx = rendered.index("Input:\nMaria is 29.")
        assert role_idx < task_idx < input_contract_idx < output_contract_idx < rules_idx < input_idx

    def test_user_input_is_always_last(self):
        template = _minimal_template(rules=["Rule A"])
        rendered = template.render("the user text")
        assert rendered.rstrip().endswith("the user text")

    def test_omits_empty_optional_sections(self):
        template = _minimal_template()  # no input/output contract, no rules
        rendered = template.render("hi")
        assert "Input contract:" not in rendered
        assert "Output contract:" not in rendered
        assert "Rules:" not in rendered

    def test_rules_are_rendered_as_a_bulleted_list(self):
        template = _minimal_template(rules=["Rule A", "Rule B"])
        rendered = template.render("hi")
        assert "- Rule A" in rendered
        assert "- Rule B" in rendered


class TestRenderExamples:
    def test_includes_few_shot_examples_when_present(self):
        template = _minimal_template(few_shot_examples=[FewShotExample(input="Bob is 40.", output='{"age": 40}')])
        rendered = template.render("hi")
        assert "Examples:" in rendered
        assert "Bob is 40." in rendered
        assert '{"age": 40}' in rendered

    def test_includes_negative_examples_when_present(self):
        template = _minimal_template(
            negative_examples=[NegativeExample(input="x", wrong_output="```json\n{}\n```", reason="no fences")]
        )
        rendered = template.render("hi")
        assert "Examples:" in rendered
        assert "no fences" in rendered

    def test_omits_examples_section_when_no_examples_given(self):
        template = _minimal_template()
        rendered = template.render("hi")
        assert "Examples:" not in rendered

    def test_includes_both_positive_and_negative_examples_together(self):
        template = _minimal_template(
            few_shot_examples=[FewShotExample(input="a", output="b")],
            negative_examples=[NegativeExample(input="c", wrong_output="d", reason="e")],
        )
        rendered = template.render("hi")
        assert rendered.count("Examples:") == 1
        assert "a" in rendered and "b" in rendered
        assert "c" in rendered and "e" in rendered


class TestInvariantPrefix:
    def test_excludes_the_final_user_input_section(self):
        template = _minimal_template(rules=["Rule A"])
        prefix = template.invariant_prefix()
        assert "Rule A" in prefix
        assert "Input:" not in prefix

    def test_is_the_same_regardless_of_user_input(self):
        template = _minimal_template(rules=["Rule A"])
        # invariant_prefix() should not depend on any particular render() call's input.
        prefix = template.invariant_prefix()
        rendered_a = template.render("input A")
        rendered_b = template.render("completely different input B")
        assert rendered_a.startswith(prefix)
        assert rendered_b.startswith(prefix)
