from PIL import Image

from local_ai_core.multimodal.vlm import FakeVLM, MlxVisionLanguageModel


def make_image() -> Image.Image:
    return Image.new("RGB", (10, 10), color="white")


def fake_load_fn(model_id: str):
    return (f"model:{model_id}", f"processor:{model_id}")


def fake_describe_fn(model, image: Image.Image, prompt: str) -> str:
    return f"described a {image.size} image for prompt {prompt!r}"


class TestFakeVLM:
    async def test_returns_the_default_response(self):
        vlm = FakeVLM(default_response="a red square")
        result = await vlm.describe(make_image(), "what is this?")
        assert result == "a red square"

    async def test_records_call_history(self):
        vlm = FakeVLM()
        image = make_image()
        await vlm.describe(image, "describe this")
        assert vlm.calls == [(image, "describe this")]

    async def test_multiple_calls_accumulate(self):
        vlm = FakeVLM()
        await vlm.describe(make_image(), "q1")
        await vlm.describe(make_image(), "q2")
        assert len(vlm.calls) == 2


class TestMlxVisionLanguageModel:
    async def test_describe_uses_the_injected_fakes(self):
        vlm = MlxVisionLanguageModel("fake-model-id", load_fn=fake_load_fn, describe_fn=fake_describe_fn)
        result = await vlm.describe(make_image(), "what is this?")
        assert "described a (10, 10) image" in result

    async def test_the_model_is_loaded_only_once_across_calls(self):
        load_calls = []

        def counting_load_fn(model_id):
            load_calls.append(model_id)
            return fake_load_fn(model_id)

        vlm = MlxVisionLanguageModel("fake-model-id", load_fn=counting_load_fn, describe_fn=fake_describe_fn)
        await vlm.describe(make_image(), "q1")
        await vlm.describe(make_image(), "q2")
        assert load_calls == ["fake-model-id"]
