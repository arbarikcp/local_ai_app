import compare_truncation_strategies as sut


class TestBuildConversationWithEarlyFact:
    def test_early_fact_is_the_second_turn(self):
        turns = sut.build_conversation_with_early_fact(5)
        assert turns[1].content == sut.EARLY_FACT

    def test_includes_the_requested_number_of_filler_turns(self):
        turns = sut.build_conversation_with_early_fact(10)
        # system + fact + ack + 10 filler turns
        assert len(turns) == 13


class TestFactAppears:
    def test_true_when_marker_present_in_any_turn(self):
        turns = sut.build_conversation_with_early_fact(2)
        assert sut.fact_appears(turns, sut.EARLY_FACT_MARKER) is True

    def test_false_when_marker_absent(self):
        turns = sut.build_conversation_with_early_fact(2)[2:]  # drop the fact turn
        assert sut.fact_appears(turns, "NOT-PRESENT-MARKER") is False


class TestCrudeSummarizeFn:
    def test_preserves_content_before_the_first_period(self):
        from local_ai_core.conversation.turn import Turn

        summary = sut.crude_summarize_fn([Turn(role="user", content="Hello there. Extra stuff.")])
        assert "Hello there" in summary

    def test_joins_multiple_turns(self):
        from local_ai_core.conversation.turn import Turn

        summary = sut.crude_summarize_fn([Turn(role="user", content="A."), Turn(role="user", content="B.")])
        assert "A" in summary and "B" in summary


class TestRunLab:
    def test_drop_oldest_loses_the_early_fact_with_enough_filler(self):
        result = sut.run_lab(n_turns=30, context_window=1500)
        assert result["drop_oldest_fact_present"] is False

    def test_summarize_then_truncate_retains_the_early_fact(self):
        result = sut.run_lab(n_turns=30, context_window=1500)
        assert result["summarize_fact_present"] is True

    def test_summarize_strategy_uses_fewer_or_equal_turns_than_original(self):
        result = sut.run_lab(n_turns=30, context_window=1500)
        assert result["summarize_turn_count"] <= result["original_turn_count"]


class TestResultToMarkdown:
    def test_renders_both_strategies_and_the_fact(self):
        result = {
            "drop_oldest_turn_count": 19, "drop_oldest_fact_present": False,
            "summarize_turn_count": 4, "summarize_fact_present": True,
        }
        md = sut.result_to_markdown(result)
        assert "drop_oldest" in md
        assert "summarize_then_truncate" in md
        assert sut.EARLY_FACT in md
