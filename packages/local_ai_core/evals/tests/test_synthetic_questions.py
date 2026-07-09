from local_ai_core.evals.synthetic_questions import generate_questions_from_document
from local_ai_core.runtimes.fake import FakeRuntime


class TestGenerateQuestionsFromDocument:
    async def test_parses_one_question_per_line(self):
        runtime = FakeRuntime(
            default_response="How long does a reset link last?\nWhat happens after 15 minutes?\nWho can request a reset?"
        )
        questions = await generate_questions_from_document("some document text", runtime, model="fake-model", n=3)
        assert len(questions) == 3

    async def test_truncates_to_n_even_if_the_model_returns_more_lines(self):
        runtime = FakeRuntime(default_response="Q1?\nQ2?\nQ3?\nQ4?")
        questions = await generate_questions_from_document("doc", runtime, model="fake-model", n=2)
        assert len(questions) == 2

    async def test_empty_response_returns_empty_list(self):
        runtime = FakeRuntime(default_response="   \n  ")
        questions = await generate_questions_from_document("doc", runtime, model="fake-model", n=3)
        assert questions == []

    async def test_sends_the_configured_model(self):
        runtime = FakeRuntime(default_response="Q1?")
        await generate_questions_from_document("doc", runtime, model="fake-model", n=1)
        assert runtime.requests_received[0].model == "fake-model"
