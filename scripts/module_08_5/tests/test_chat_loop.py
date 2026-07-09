import chat_loop as sut
from local_ai_core.conversation.session_store import SessionStore
from local_ai_core.runtimes.fake import FakeRuntime


class TestIsForgetCommand:
    def test_matches_the_exact_command(self):
        assert sut.is_forget_command("/forget") is True

    def test_matches_with_surrounding_whitespace(self):
        assert sut.is_forget_command("  /forget  ") is True

    def test_does_not_match_other_text(self):
        assert sut.is_forget_command("please forget everything") is False

    def test_does_not_match_partial_command(self):
        assert sut.is_forget_command("/forg") is False


class TestRenderHistory:
    def test_renders_role_and_content_for_each_turn(self):
        from local_ai_core.conversation.turn import Turn

        history = [Turn(role="system", content="be terse"), Turn(role="user", content="hi")]
        rendered = sut.render_history(history)
        assert "system: be terse" in rendered
        assert "user: hi" in rendered

    def test_empty_history_renders_empty_string(self):
        assert sut.render_history([]) == ""


class TestProcessUserInput:
    async def test_appends_user_and_assistant_turns(self):
        store = SessionStore()
        runtime = FakeRuntime(default_response="a response")
        await sut.process_user_input(store, "s1", "hello", runtime, "m")
        turns = store.get_turns("s1")
        assert [t.role for t in turns] == ["user", "assistant"]
        assert turns[0].content == "hello"
        assert turns[1].content == "a response"

    async def test_adds_a_sticky_system_prompt_on_first_call(self):
        store = SessionStore()
        runtime = FakeRuntime(default_response="ok")
        await sut.process_user_input(store, "s1", "hi", runtime, "m", system_prompt="Be helpful.")
        turns = store.get_turns("s1")
        assert turns[0].role == "system"
        assert turns[0].sticky is True
        assert turns[0].content == "Be helpful."

    async def test_does_not_re_add_system_prompt_on_subsequent_calls(self):
        store = SessionStore()
        runtime = FakeRuntime(default_response="ok")
        await sut.process_user_input(store, "s1", "first", runtime, "m", system_prompt="Be helpful.")
        await sut.process_user_input(store, "s1", "second", runtime, "m", system_prompt="Be helpful.")
        system_turns = [t for t in store.get_turns("s1") if t.role == "system"]
        assert len(system_turns) == 1

    async def test_passes_full_history_to_the_runtime(self):
        received_prompts = []

        class RecordingRuntime(FakeRuntime):
            async def generate(self, request):
                received_prompts.append(request.prompt)
                return await super().generate(request)

        store = SessionStore()
        runtime = RecordingRuntime(default_response="ok")
        await sut.process_user_input(store, "s1", "first message", runtime, "m")
        await sut.process_user_input(store, "s1", "second message", runtime, "m")
        # Second call's prompt should include the first exchange too.
        assert "first message" in received_prompts[1]
        assert "second message" in received_prompts[1]

    async def test_returns_the_assistant_turn(self):
        store = SessionStore()
        runtime = FakeRuntime(default_response="the answer")
        turn = await sut.process_user_input(store, "s1", "question", runtime, "m")
        assert turn.role == "assistant"
        assert turn.content == "the answer"


class TestMainSkipPath:
    def test_main_skips_cleanly_when_ollama_unreachable(self, capsys):
        exit_code = sut.main(["--model", "qwen2.5:1.5b"])
        assert exit_code == 1
        assert "SKIPPED" in capsys.readouterr().err
