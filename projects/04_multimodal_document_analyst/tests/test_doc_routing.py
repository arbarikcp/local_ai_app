import pytest
from PIL import Image

from doc_routing import decide_route
from local_ai_core.multimodal.routing import MultimodalRoute


class TestDecideRoute:
    def test_a_usable_text_layer_routes_to_text_llm_without_image_math(self):
        decision = decide_route("A" * 200, context_window=4096)
        assert decision.route == MultimodalRoute.TEXT_LLM
        assert decision.image_tokens is None
        assert decision.context_budget_fraction is None

    def test_no_text_layer_routes_to_vlm_and_computes_real_image_cost(self):
        image = Image.new("RGB", (400, 300))
        decision = decide_route("", image=image, context_window=4096)
        assert decision.route == MultimodalRoute.VLM
        assert decision.image_tokens == 29 * 22  # ceil(400/14) * ceil(300/14)
        assert decision.context_budget_fraction == pytest.approx(decision.image_tokens / 4096)
        assert "tokens" in decision.reason

    def test_vlm_route_without_an_image_raises(self):
        with pytest.raises(ValueError):
            decide_route("", image=None, context_window=4096)

    def test_a_sparse_text_layer_below_threshold_routes_to_vlm(self):
        image = Image.new("RGB", (100, 100))
        decision = decide_route("only a few words", image=image, context_window=4096)
        assert decision.route == MultimodalRoute.VLM

    def test_custom_min_text_chars_is_honored(self):
        decision = decide_route("short", context_window=4096, min_text_chars=3)
        assert decision.route == MultimodalRoute.TEXT_LLM

    def test_a_larger_image_produces_a_larger_context_budget_fraction(self):
        small = decide_route("", image=Image.new("RGB", (100, 100)), context_window=4096)
        large = decide_route("", image=Image.new("RGB", (2000, 2000)), context_window=4096)
        assert large.context_budget_fraction > small.context_budget_fraction
