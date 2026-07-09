from local_ai_core.conversation.turn import Turn, to_chat_messages


class TestTurn:
    def test_minimal_turn_has_sane_defaults(self):
        turn = Turn(role="user", content="hi")
        assert turn.tool_call_id is None
        assert turn.turn_group_id is None
        assert turn.sticky is False

    def test_full_turn_round_trips(self):
        turn = Turn(role="tool", content="result", tool_call_id="call-1", turn_group_id="group-1", sticky=True)
        assert turn.tool_call_id == "call-1"
        assert turn.turn_group_id == "group-1"
        assert turn.sticky is True

    def test_turn_is_immutable(self):
        import pytest
        from dataclasses import FrozenInstanceError

        turn = Turn(role="user", content="hi")
        with pytest.raises(FrozenInstanceError):
            turn.content = "changed"


class TestToChatMessages:
    def test_converts_role_and_content_only(self):
        turns = [Turn(role="system", content="be terse"), Turn(role="user", content="hi")]
        messages = to_chat_messages(turns)
        assert messages == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]

    def test_omits_tool_call_id_and_sticky_from_output(self):
        turns = [Turn(role="tool", content="result", tool_call_id="call-1", sticky=True)]
        messages = to_chat_messages(turns)
        assert messages == [{"role": "tool", "content": "result"}]

    def test_empty_list_returns_empty_list(self):
        assert to_chat_messages([]) == []
