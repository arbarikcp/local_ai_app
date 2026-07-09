import force_past_context_window as sut


class TestBuildSyntheticConversation:
    def test_produces_n_turns_plus_a_sticky_system_turn(self):
        turns = sut.build_synthetic_conversation(10)
        assert len(turns) == 11
        assert turns[0].sticky is True

    def test_alternates_user_and_assistant_roles(self):
        turns = sut.build_synthetic_conversation(4)
        roles = [t.role for t in turns[1:]]  # skip the system turn
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_more_words_per_turn_means_longer_content(self):
        short = sut.build_synthetic_conversation(1, words_per_turn=5)
        long = sut.build_synthetic_conversation(1, words_per_turn=50)
        assert len(long[1].content) > len(short[1].content)


class TestRunLab:
    def test_exceeds_budget_before_truncation_with_enough_turns(self):
        result = sut.run_lab(n_turns=40, context_window=2000)
        assert result["exceeded_before_truncation"] is True

    def test_does_not_exceed_budget_after_truncation(self):
        result = sut.run_lab(n_turns=40, context_window=2000)
        assert result["exceeded_after_truncation"] is False

    def test_truncation_actually_reduces_turn_count(self):
        result = sut.run_lab(n_turns=40, context_window=2000)
        assert result["truncated_turn_count"] < result["original_turn_count"]

    def test_sticky_system_prompt_survives_truncation(self):
        result = sut.run_lab(n_turns=40, context_window=2000)
        assert result["system_prompt_retained"] is True

    def test_small_conversation_never_exceeds_a_generous_budget(self):
        result = sut.run_lab(n_turns=2, context_window=100_000)
        assert result["exceeded_before_truncation"] is False
        assert result["truncated_turn_count"] == result["original_turn_count"]


class TestResultToMarkdown:
    def test_renders_all_fields(self):
        result = {
            "original_turn_count": 41, "original_token_estimate": 2710, "history_budget": 1500,
            "exceeded_before_truncation": True, "truncated_turn_count": 23,
            "truncated_token_estimate": 1494, "exceeded_after_truncation": False,
            "system_prompt_retained": True,
        }
        md = sut.result_to_markdown(result)
        assert "41" in md and "23" in md
        assert "1500 tokens" in md
