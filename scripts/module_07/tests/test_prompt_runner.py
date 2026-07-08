import pytest

import prompt_runner as sut
from local_ai_core.runtimes.fake import FakeRuntime


class TestRunVariantAgainstModel:
    async def test_collects_one_output_per_test_input(self):
        runtime = FakeRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        result = await sut.run_variant_against_model("v4_with_schema", "fake-model", runtime, ["a", "b", "c"])
        assert len(result.outputs) == 3
        assert result.variant_name == "v4_with_schema"
        assert result.model == "fake-model"

    async def test_invalid_json_rate_is_zero_for_valid_json_responses(self):
        runtime = FakeRuntime(default_response='{"name": "X"}')
        result = await sut.run_variant_against_model("v4_with_schema", "m", runtime, ["a"])
        assert result.invalid_json_rate == 0.0

    async def test_invalid_json_rate_is_one_for_non_json_responses(self):
        runtime = FakeRuntime(default_response="not json at all")
        result = await sut.run_variant_against_model("v1_vague", "m", runtime, ["a"])
        assert result.invalid_json_rate == 1.0

    async def test_invalid_json_rate_reflects_a_mix(self):
        # Alternate valid/invalid responses using a per-model response map is
        # awkward for a single model, so use a runtime whose response
        # depends on call count via a custom fake generate approach instead.
        responses = iter(['{"a": 1}', "not json", '{"b": 2}', "still not json"])

        class AlternatingRuntime(FakeRuntime):
            async def generate(self, request):
                self.responses = {request.model: next(responses)}
                return await super().generate(request)

        runtime = AlternatingRuntime()
        result = await sut.run_variant_against_model("v1_vague", "m", runtime, ["a", "b", "c", "d"])
        assert result.invalid_json_rate == pytest.approx(0.5)

    async def test_renders_the_correct_variant_for_each_input(self):
        received_prompts = []

        class RecordingRuntime(FakeRuntime):
            async def generate(self, request):
                received_prompts.append(request.prompt)
                return await super().generate(request)

        runtime = RecordingRuntime(default_response="{}")
        await sut.run_variant_against_model("v1_vague", "m", runtime, ["unique input text"])
        assert "unique input text" in received_prompts[0]
        assert "Get the important info from this" in received_prompts[0]


class TestRunLab:
    async def test_produces_one_result_per_variant_per_model(self):
        runtime = FakeRuntime(default_response="{}")
        results = await sut.run_lab(runtime, ["model-a", "model-b"], test_inputs=["x"])
        assert len(results) == len(sut.ALL_VARIANTS) * 2

    async def test_covers_every_variant_name(self):
        runtime = FakeRuntime(default_response="{}")
        results = await sut.run_lab(runtime, ["model-a"], test_inputs=["x"])
        variant_names = {r.variant_name for r in results}
        assert variant_names == set(sut.ALL_VARIANTS.keys())

    async def test_uses_default_test_inputs_when_none_given(self):
        runtime = FakeRuntime(default_response="{}")
        results = await sut.run_lab(runtime, ["model-a"])
        assert len(results[0].outputs) == len(sut.DEFAULT_TEST_INPUTS)


class TestResultsToMarkdownTable:
    def test_renders_all_results(self):
        results = [
            sut.VariantResult(variant_name="v1_vague", model="m1", outputs=["a"], invalid_json_rate=1.0),
            sut.VariantResult(variant_name="v5_with_few_shot", model="m1", outputs=["b"], invalid_json_rate=0.0),
        ]
        table = sut.results_to_markdown_table(results)
        assert "v1_vague" in table
        assert "v5_with_few_shot" in table
        assert "100%" in table
        assert "0%" in table


class TestMainSkipPath:
    def test_main_skips_cleanly_when_ollama_unreachable(self, capsys):
        exit_code = sut.main(["--models", "qwen2.5:1.5b"])
        assert exit_code == 1
        assert "SKIPPED" in capsys.readouterr().err
