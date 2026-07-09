from local_ai_core.multimodal.routing import MultimodalRoute, should_use_vlm


class TestShouldUseVlm:
    def test_a_document_with_a_long_text_layer_routes_to_text_llm(self):
        decision = should_use_vlm("This is a real, coherent block of extracted invoice text.")
        assert decision.route == MultimodalRoute.TEXT_LLM

    def test_an_empty_text_layer_routes_to_vlm(self):
        decision = should_use_vlm("")
        assert decision.route == MultimodalRoute.VLM

    def test_a_sparse_text_layer_below_the_threshold_routes_to_vlm(self):
        decision = should_use_vlm("short", min_text_chars=40)
        assert decision.route == MultimodalRoute.VLM

    def test_the_threshold_is_adjustable(self):
        decision = should_use_vlm("exactly ten", min_text_chars=5)
        assert decision.route == MultimodalRoute.TEXT_LLM

    def test_whitespace_only_text_counts_as_empty(self):
        decision = should_use_vlm("   \n\t  ")
        assert decision.route == MultimodalRoute.VLM

    def test_the_reason_explains_the_decision(self):
        decision = should_use_vlm("")
        assert "0 chars" in decision.reason
