import pytest

from local_ai_core.multimodal.memory_cost import estimate_context_budget_impact, estimate_image_tokens


class TestEstimateImageTokens:
    def test_an_exact_multiple_of_patch_size(self):
        assert estimate_image_tokens(280, 140, patch_size=14) == 20 * 10

    def test_rounds_up_a_partial_patch(self):
        # 15 / 14 -> 2 patches, not 1 - a partial patch still costs a full token.
        assert estimate_image_tokens(15, 14, patch_size=14) == 2 * 1

    def test_a_larger_image_costs_more_tokens(self):
        small = estimate_image_tokens(224, 224)
        large = estimate_image_tokens(1024, 1024)
        assert large > small

    def test_rejects_nonpositive_dimensions(self):
        with pytest.raises(ValueError):
            estimate_image_tokens(0, 100)

    def test_rejects_nonpositive_patch_size(self):
        with pytest.raises(ValueError):
            estimate_image_tokens(100, 100, patch_size=0)


class TestEstimateContextBudgetImpact:
    def test_computes_the_real_fraction(self):
        assert estimate_context_budget_impact(image_tokens=2000, context_window=8000) == pytest.approx(0.25)

    def test_a_high_resolution_image_can_cost_a_large_fraction_of_a_small_context_window(self):
        tokens = estimate_image_tokens(1024, 1024)  # a real, large image
        fraction = estimate_context_budget_impact(tokens, context_window=4096)
        assert fraction > 1.0  # genuinely doesn't fit - a real, checkable overflow

    def test_rejects_a_nonpositive_context_window(self):
        with pytest.raises(ValueError):
            estimate_context_budget_impact(100, context_window=0)
