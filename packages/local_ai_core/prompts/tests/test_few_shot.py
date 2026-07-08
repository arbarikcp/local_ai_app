from local_ai_core.prompts.few_shot import (
    FewShotExample,
    NegativeExample,
    format_few_shot_examples,
    format_negative_examples,
)


class TestFormatFewShotExamples:
    def test_empty_list_returns_empty_string(self):
        assert format_few_shot_examples([]) == ""

    def test_single_example_includes_input_and_output(self):
        result = format_few_shot_examples([FewShotExample(input="hi", output='{"a": 1}')])
        assert "hi" in result
        assert '{"a": 1}' in result

    def test_multiple_examples_are_separated(self):
        examples = [
            FewShotExample(input="in1", output="out1"),
            FewShotExample(input="in2", output="out2"),
        ]
        result = format_few_shot_examples(examples)
        assert "in1" in result
        assert "in2" in result
        assert result.count("Input:") == 2


class TestFormatNegativeExamples:
    def test_empty_list_returns_empty_string(self):
        assert format_negative_examples([]) == ""

    def test_includes_input_wrong_output_and_reason(self):
        result = format_negative_examples(
            [NegativeExample(input="hi", wrong_output="```json\n{}\n```", reason="no markdown fences allowed")]
        )
        assert "hi" in result
        assert "```json" in result
        assert "no markdown fences allowed" in result

    def test_multiple_negative_examples_are_separated(self):
        examples = [
            NegativeExample(input="in1", wrong_output="bad1", reason="reason1"),
            NegativeExample(input="in2", wrong_output="bad2", reason="reason2"),
        ]
        result = format_negative_examples(examples)
        assert result.count("Why:") == 2
