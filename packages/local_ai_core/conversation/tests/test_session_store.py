from local_ai_core.conversation.session_store import SessionStore
from local_ai_core.conversation.turn import Turn


class TestCreateSession:
    def test_generates_a_session_id_when_none_given(self):
        store = SessionStore()
        session_id = store.create_session()
        assert session_id

    def test_accepts_a_caller_chosen_session_id(self):
        store = SessionStore()
        session_id = store.create_session("my-session")
        assert session_id == "my-session"


class TestAppendAndGetTurns:
    def test_append_then_get_returns_the_turn(self):
        store = SessionStore()
        session_id = store.create_session()
        store.append_turn(session_id, Turn(role="user", content="hello"))
        turns = store.get_turns(session_id)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "hello"

    def test_turns_are_returned_in_insertion_order(self):
        store = SessionStore()
        session_id = store.create_session()
        store.append_turn(session_id, Turn(role="user", content="first"))
        store.append_turn(session_id, Turn(role="assistant", content="second"))
        store.append_turn(session_id, Turn(role="user", content="third"))
        turns = store.get_turns(session_id)
        assert [t.content for t in turns] == ["first", "second", "third"]

    def test_preserves_tool_call_id_and_turn_group_id(self):
        store = SessionStore()
        session_id = store.create_session()
        store.append_turn(
            session_id, Turn(role="tool", content="result", tool_call_id="call-1", turn_group_id="g1")
        )
        turns = store.get_turns(session_id)
        assert turns[0].tool_call_id == "call-1"
        assert turns[0].turn_group_id == "g1"

    def test_preserves_sticky_flag(self):
        store = SessionStore()
        session_id = store.create_session()
        store.append_turn(session_id, Turn(role="system", content="be terse", sticky=True))
        turns = store.get_turns(session_id)
        assert turns[0].sticky is True

    def test_different_sessions_are_isolated(self):
        store = SessionStore()
        s1 = store.create_session("s1")
        s2 = store.create_session("s2")
        store.append_turn(s1, Turn(role="user", content="in session 1"))
        store.append_turn(s2, Turn(role="user", content="in session 2"))
        assert [t.content for t in store.get_turns(s1)] == ["in session 1"]
        assert [t.content for t in store.get_turns(s2)] == ["in session 2"]

    def test_get_turns_for_unknown_session_returns_empty_list(self):
        store = SessionStore()
        assert store.get_turns("never-created") == []


class TestListSessions:
    def test_empty_store_has_no_sessions(self):
        assert SessionStore().list_sessions() == []

    def test_lists_every_session_with_at_least_one_turn(self):
        store = SessionStore()
        store.append_turn("s1", Turn(role="user", content="a"))
        store.append_turn("s2", Turn(role="user", content="b"))
        assert set(store.list_sessions()) == {"s1", "s2"}


class TestSessionExists:
    def test_false_for_a_session_with_no_turns(self):
        assert SessionStore().session_exists("never-created") is False

    def test_true_after_appending_a_turn(self):
        store = SessionStore()
        store.append_turn("s1", Turn(role="user", content="a"))
        assert store.session_exists("s1") is True


class TestDeleteSession:
    def test_removes_all_turns_for_the_session(self):
        store = SessionStore()
        store.append_turn("s1", Turn(role="user", content="a"))
        store.append_turn("s1", Turn(role="user", content="b"))
        store.delete_session("s1")
        assert store.get_turns("s1") == []
        assert store.session_exists("s1") is False

    def test_does_not_affect_other_sessions(self):
        store = SessionStore()
        store.append_turn("s1", Turn(role="user", content="a"))
        store.append_turn("s2", Turn(role="user", content="b"))
        store.delete_session("s1")
        assert store.get_turns("s2") == [Turn(role="user", content="b")]

    def test_deleting_a_nonexistent_session_does_not_raise(self):
        store = SessionStore()
        store.delete_session("never-created")  # must not raise


class TestResumptionAfterRestart:
    def test_turns_persist_across_a_real_close_and_reopen_cycle(self, tmp_path):
        db_path = tmp_path / "sessions.db"

        store1 = SessionStore(db_path)
        session_id = store1.create_session("persisted-session")
        store1.append_turn(session_id, Turn(role="user", content="before restart"))
        store1.append_turn(session_id, Turn(role="assistant", content="response before restart"))
        store1.close()

        # A genuinely new SessionStore instance, same file path - simulates
        # the application restarting.
        store2 = SessionStore(db_path)
        turns = store2.get_turns(session_id)
        assert [t.content for t in turns] == ["before restart", "response before restart"]
        store2.close()

    def test_can_continue_appending_after_reopening(self, tmp_path):
        db_path = tmp_path / "sessions.db"

        store1 = SessionStore(db_path)
        store1.append_turn("s1", Turn(role="user", content="turn 1"))
        store1.close()

        store2 = SessionStore(db_path)
        store2.append_turn("s1", Turn(role="assistant", content="turn 2"))
        turns = store2.get_turns("s1")
        assert [t.content for t in turns] == ["turn 1", "turn 2"]
        store2.close()


class TestContextManager:
    def test_can_be_used_as_a_context_manager(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        with SessionStore(db_path) as store:
            store.append_turn("s1", Turn(role="user", content="hi"))
            assert store.get_turns("s1")[0].content == "hi"
        # connection is closed after the with block; reopening should still see the data
        with SessionStore(db_path) as store2:
            assert store2.get_turns("s1")[0].content == "hi"
