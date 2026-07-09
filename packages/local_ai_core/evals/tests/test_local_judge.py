import pytest

from local_ai_core.evals.local_judge import JudgeParseError, LocalJudge
from local_ai_core.runtimes.fake import FakeRuntime


class TestJudge:
    async def test_parses_a_faithful_verdict(self):
        runtime = FakeRuntime(default_response='{"faithful": true, "reasoning": "Answer matches context."}')
        judge = LocalJudge(runtime, model="fake-model")
        verdict = await judge.judge("Q?", "context", "answer")
        assert verdict.faithful is True
        assert verdict.reasoning == "Answer matches context."

    async def test_parses_an_unfaithful_verdict(self):
        runtime = FakeRuntime(default_response='{"faithful": false, "reasoning": "Not supported."}')
        judge = LocalJudge(runtime, model="fake-model")
        verdict = await judge.judge("Q?", "context", "answer")
        assert verdict.faithful is False

    async def test_raises_on_unparseable_response(self):
        runtime = FakeRuntime(default_response="not json at all")
        judge = LocalJudge(runtime, model="fake-model")
        with pytest.raises(JudgeParseError):
            await judge.judge("Q?", "context", "answer")

    async def test_raises_when_faithful_key_is_missing(self):
        runtime = FakeRuntime(default_response='{"reasoning": "no faithful key"}')
        judge = LocalJudge(runtime, model="fake-model")
        with pytest.raises(JudgeParseError):
            await judge.judge("Q?", "context", "answer")

    async def test_sends_the_configured_model(self):
        runtime = FakeRuntime(default_response='{"faithful": true, "reasoning": "ok"}')
        judge = LocalJudge(runtime, model="fake-model")
        await judge.judge("Q?", "context", "answer")
        assert runtime.requests_received[0].model == "fake-model"
