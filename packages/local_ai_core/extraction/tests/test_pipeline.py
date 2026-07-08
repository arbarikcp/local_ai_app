from local_ai_core.extraction.pipeline import ExtractionPipeline, build_extraction_prompt, build_repair_prompt
from local_ai_core.extraction.review_queue import ReviewQueue
from local_ai_core.extraction.schemas import PersonExtraction
from local_ai_core.runtimes.errors import FeatureNotSupported
from local_ai_core.runtimes.fake import FakeRuntime


class SequencedRuntime(FakeRuntime):
    """Returns responses from a fixed sequence, one per call."""

    def __init__(self, responses: list[str], **kwargs):
        super().__init__(**kwargs)
        self._sequence = iter(responses)

    async def generate(self, request):
        text = next(self._sequence)
        self.responses = {request.model: text}
        return await super().generate(request)


class NoJsonSchemaRuntime(FakeRuntime):
    """Simulates an adapter (like MLXRuntime) with no structured-output support."""

    async def generate(self, request):
        if request.response_format.type == "json_schema":
            raise FeatureNotSupported("no json_schema support")
        return await super().generate(request)


# --- Pure prompt-building functions -----------------------------------------


class TestBuildExtractionPrompt:
    def test_includes_the_input_text(self):
        prompt = build_extraction_prompt("Maria is 29.", PersonExtraction)
        assert "Maria is 29." in prompt

    def test_includes_the_schema(self):
        prompt = build_extraction_prompt("text", PersonExtraction)
        assert "name" in prompt and "age" in prompt and "city" in prompt

    def test_includes_the_curriculum_rules(self):
        prompt = build_extraction_prompt("text", PersonExtraction)
        assert "Return only valid JSON" in prompt
        assert "Do not include markdown" in prompt


class TestBuildRepairPrompt:
    def test_includes_original_prompt_invalid_output_and_error(self):
        repaired = build_repair_prompt("original prompt text", "```json\n{}\n```", "missing required field")
        assert "original prompt text" in repaired
        assert "```json" in repaired
        assert "missing required field" in repaired


# --- ExtractionPipeline.run() -----------------------------------------------


class TestRunSuccessfulExtraction:
    async def test_valid_response_produces_a_parsed_model_and_high_confidence(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        result = await pipeline.run("Maria moved to Austin. She is 29.", "fake-model")
        assert result.parsed is not None
        assert result.parsed.name == "Maria"
        assert result.confidence == "high"
        assert result.needs_review is False
        assert result.used_constrained_decoding is True
        assert result.used_repair_retry is False

    async def test_high_confidence_result_is_not_enqueued_for_review(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        queue = ReviewQueue()
        pipeline = ExtractionPipeline(
            runtime, PersonExtraction, required_fields=["name", "age", "city"], review_queue=queue
        )
        await pipeline.run("text", "m")
        assert len(queue) == 0

    async def test_requests_json_schema_response_format(self):
        received_formats = []

        class RecordingRuntime(FakeRuntime):
            async def generate(self, request):
                received_formats.append(request.response_format.type)
                return await super().generate(request)

        runtime = RecordingRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction)
        await pipeline.run("text", "m")
        assert received_formats[0] == "json_schema"


class TestRunFallbackOnFeatureNotSupported:
    async def test_falls_back_to_prompt_only_and_records_it(self):
        runtime = NoJsonSchemaRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        result = await pipeline.run("text", "m")
        assert result.used_constrained_decoding is False
        assert result.parsed is not None  # fallback still succeeded

    async def test_fallback_with_bad_output_downgrades_confidence(self):
        runtime = NoJsonSchemaRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        result = await pipeline.run("text", "m")
        # unconstrained decoding alone is one risk factor -> medium, not high
        assert result.confidence == "medium"


class TestRunRepairRetry:
    async def test_invalid_first_response_then_valid_repair_succeeds(self):
        runtime = SequencedRuntime(["not valid json at all", '{"name": "Maria", "age": 29, "city": "Austin"}'])
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        result = await pipeline.run("text", "m")
        assert result.parsed is not None
        assert result.used_repair_retry is True
        assert result.confidence == "medium"  # repair retry is one risk factor

    async def test_exhausting_repair_attempts_leaves_result_unparsed_and_flagged(self):
        runtime = SequencedRuntime(["not json", "still not json"])
        queue = ReviewQueue()
        pipeline = ExtractionPipeline(
            runtime, PersonExtraction, required_fields=["name", "age", "city"], max_repair_attempts=1, review_queue=queue
        )
        result = await pipeline.run("text", "m")
        assert result.parsed is None
        assert result.needs_review is True
        assert len(queue) == 1

    async def test_repair_attempts_respect_the_configured_max(self):
        call_count = {"n": 0}

        class CountingSequencedRuntime(SequencedRuntime):
            async def generate(self, request):
                call_count["n"] += 1
                return await super().generate(request)

        runtime = CountingSequencedRuntime(["bad"] * 5)
        pipeline = ExtractionPipeline(runtime, PersonExtraction, max_repair_attempts=2)
        await pipeline.run("text", "m")
        assert call_count["n"] == 3  # 1 initial + 2 repair attempts


class TestRunMissingRequiredFields:
    async def test_missing_field_downgrades_confidence_and_queues_for_review(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": null, "city": null}')
        queue = ReviewQueue()
        pipeline = ExtractionPipeline(
            runtime, PersonExtraction, required_fields=["name", "age", "city"], review_queue=queue
        )
        result = await pipeline.run("Maria is here.", "m")
        assert result.confidence == "medium"  # exactly one risk factor: missing required field
        assert result.needs_review is False  # medium doesn't trigger review by default (only "low" does)
        assert len(queue) == 0


# --- ExtractionPipeline.run_chunked() ---------------------------------------


class TestRunChunked:
    async def test_short_text_delegates_directly_to_run(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        result = await pipeline.run_chunked("short text", "m", max_chars=10_000)
        assert result.parsed is not None
        assert runtime.call_count == 1  # no chunking needed, one generate() call total

    async def test_long_text_is_chunked_and_merged(self):
        runtime = FakeRuntime(default_response='{"name": "Maria", "age": 29, "city": "Austin"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
        long_text = ("Paragraph about Maria. " * 20 + "\n\n") * 5
        result = await pipeline.run_chunked(long_text, "m", max_chars=200)
        assert runtime.call_count > 1  # multiple chunks were processed
        assert result.fields["name"] == "Maria"

    async def test_conflicting_chunk_values_flag_for_review(self):
        # Two chunks disagree on "city" - the runtime returns different
        # values depending on call count.
        responses = iter(
            [
                '{"name": "Maria", "age": 29, "city": "Austin"}',
                '{"name": "Maria", "age": 29, "city": "Denver"}',
            ]
        )

        class ConflictingRuntime(FakeRuntime):
            async def generate(self, request):
                self.responses = {request.model: next(responses)}
                return await super().generate(request)

        runtime = ConflictingRuntime()
        queue = ReviewQueue()
        pipeline = ExtractionPipeline(
            runtime, PersonExtraction, required_fields=["name", "age", "city"], review_queue=queue
        )
        # Force exactly 2 chunks via a long text with a clean paragraph break.
        text = ("A" * 50) + "\n\n" + ("B" * 50)
        result = await pipeline.run_chunked(text, "m", max_chars=50)
        assert "city" in result.fields  # first value kept
        assert result.needs_review is True
        assert len(queue) == 1


# --- response_format_type selection (Lab 8's 3-way comparison hook) --------


class TestResponseFormatTypeSelection:
    async def test_json_schema_mode_requests_json_schema_and_counts_as_constrained(self):
        received = []

        class RecordingRuntime(FakeRuntime):
            async def generate(self, request):
                received.append(request.response_format.type)
                return await super().generate(request)

        runtime = RecordingRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, response_format_type="json_schema")
        result = await pipeline.run("text", "m")
        assert received[0] == "json_schema"
        assert result.used_constrained_decoding is True

    async def test_grammar_mode_requests_grammar_and_counts_as_constrained(self):
        # Plain FakeRuntime deliberately does NOT support grammar (Module 6's
        # fake.py, matching real Ollama/MLX capability) - use a runtime that
        # actually supports it (matching real llama.cpp capability) to test
        # the success path, and see
        # test_grammar_mode_falls_back_when_runtime_lacks_grammar_support
        # below for the (much more common, on this course's target runtimes)
        # unsupported path.
        received = []

        class GrammarSupportingRuntime(FakeRuntime):
            def _next_error(self, request):
                if request.response_format.type == "grammar":
                    return None  # override: this fake DOES support grammar
                return super()._next_error(request)

            async def generate(self, request):
                received.append(request.response_format.type)
                return await super().generate(request)

        runtime = GrammarSupportingRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, response_format_type="grammar")
        result = await pipeline.run("text", "m")
        assert received[0] == "grammar"
        assert result.used_constrained_decoding is True

    async def test_text_mode_requests_no_structured_format_and_is_never_marked_constrained(self):
        received = []

        class RecordingRuntime(FakeRuntime):
            async def generate(self, request):
                received.append(request.response_format.type)
                return await super().generate(request)

        runtime = RecordingRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, response_format_type="text")
        result = await pipeline.run("text", "m")
        assert received[0] == "text"
        assert result.used_constrained_decoding is False

    async def test_grammar_mode_falls_back_when_runtime_lacks_grammar_support(self):
        class NoGrammarRuntime(FakeRuntime):
            async def generate(self, request):
                if request.response_format.type == "grammar":
                    raise FeatureNotSupported("no grammar support")
                return await super().generate(request)

        runtime = NoGrammarRuntime(default_response='{"name": "X", "age": 1, "city": "Y"}')
        pipeline = ExtractionPipeline(runtime, PersonExtraction, response_format_type="grammar")
        result = await pipeline.run("text", "m")
        assert result.used_constrained_decoding is False
        assert result.parsed is not None  # fallback still succeeded


class TestPlaceholderGbnfGrammar:
    def test_mentions_every_schema_field(self):
        from local_ai_core.extraction.pipeline import placeholder_gbnf_grammar

        grammar = placeholder_gbnf_grammar(PersonExtraction)
        assert "name" in grammar and "age" in grammar and "city" in grammar

    def test_is_labeled_as_a_placeholder_not_production_grammar(self):
        from local_ai_core.extraction.pipeline import placeholder_gbnf_grammar

        grammar = placeholder_gbnf_grammar(PersonExtraction)
        assert "placeholder" in grammar.lower()
