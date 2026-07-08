import pytest

from lab_1_2_long_prompt_stress_test import build_prompt
from token_estimate import estimate_tokens_heuristic


@pytest.mark.parametrize("target", [500, 2_000, 4_000])
def test_build_prompt_is_roughly_target_length(target):
    prompt = build_prompt(target)
    estimated = estimate_tokens_heuristic(prompt)
    # Allow slack: the question/instruction text adds a small fixed overhead.
    assert estimated == pytest.approx(target, rel=0.15)


def test_build_prompt_contains_the_fixed_question():
    prompt = build_prompt(500)
    assert "What is the capital of France?" in prompt


def test_build_prompt_labels_filler_as_ignorable():
    prompt = build_prompt(500)
    assert "Ignore the filler" in prompt
