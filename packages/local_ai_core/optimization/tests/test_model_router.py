from local_ai_core.optimization.model_router import ModelTier, route_model


class TestEscalationSignalsRouteToLarge:
    def test_multi_step_reasoning_routes_to_large(self):
        result = route_model(prompt_token_count=10, requires_multi_step_reasoning=True)
        assert result.tier == ModelTier.LARGE

    def test_tool_calls_route_to_large(self):
        result = route_model(prompt_token_count=10, requires_tool_calls=True)
        assert result.tier == ModelTier.LARGE

    def test_structured_output_routes_to_large(self):
        result = route_model(prompt_token_count=10, output_must_be_structured=True)
        assert result.tier == ModelTier.LARGE

    def test_long_prompt_routes_to_large(self):
        result = route_model(prompt_token_count=5000, large_model_token_threshold=2000)
        assert result.tier == ModelTier.LARGE

    def test_any_single_signal_is_enough(self):
        # Unlike Module 19's fine-tuning gate, escalation here needs only one
        # strong signal, not all of them together.
        result = route_model(prompt_token_count=10, requires_tool_calls=True)
        assert result.tier == ModelTier.LARGE


class TestNoSignalsRouteToSmall:
    def test_short_simple_prompt_routes_to_small(self):
        result = route_model(prompt_token_count=50)
        assert result.tier == ModelTier.SMALL

    def test_prompt_at_exactly_the_threshold_routes_to_small(self):
        result = route_model(prompt_token_count=2000, large_model_token_threshold=2000)
        assert result.tier == ModelTier.SMALL


class TestReasonIsTraceable:
    def test_every_decision_carries_a_nonempty_reason(self):
        for kwargs in [
            {"prompt_token_count": 10, "requires_multi_step_reasoning": True},
            {"prompt_token_count": 10},
        ]:
            result = route_model(**kwargs)
            assert len(result.reason) > 0
