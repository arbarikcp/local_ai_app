"""ExtractionPipeline — the full production extraction pipeline (theory doc
"Production extraction pipeline" and "Constrained decoding is the primary
reliability layer").

Always requests response_format.type="json_schema" first; a runtime that
raises FeatureNotSupported (Module 6) triggers an explicit, recorded
fallback to prompt-only mode rather than a silent degrade. Never calls
LLMRuntime.stream() - structured output is buffered, not streamed (theory
doc "Streaming vs structured output").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from local_ai_core.runtimes.errors import FeatureNotSupported
from local_ai_core.runtimes.types import LLMRequest, ResponseFormat

from .chunking import chunk_text, merge_partial_extractions
from .confidence import ConfidenceInputs, ConfidenceLevel, compute_confidence
from .json_parsing import try_parse_json
from .review_queue import ReviewQueue

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def build_extraction_prompt(text: str, schema: type[BaseModel]) -> str:
    """The curriculum's own example extraction prompt (theory doc), rendered
    for a given Pydantic schema.
    """
    return (
        "You are an information extraction engine.\n\n"
        "Task:\n"
        "Extract the requested fields from the input text.\n\n"
        "Rules:\n"
        "- Return only valid JSON.\n"
        "- Do not include markdown.\n"
        "- If a field is missing, use null.\n"
        "- Do not infer values that are not present.\n"
        "- Follow the schema exactly.\n\n"
        f"Schema:\n{schema.model_json_schema()}\n\n"
        f"Input:\n{text}"
    )


def build_repair_prompt(original_prompt: str, invalid_output: str, error_message: str) -> str:
    """A narrower ask than re-running the original prompt: show the model
    its own invalid output and the specific error, and ask for exactly that
    fix (theory doc §5).
    """
    return (
        f"{original_prompt}\n\n"
        "Your previous response was invalid:\n"
        f"{invalid_output}\n\n"
        f"Validation error:\n{error_message}\n\n"
        "Return ONLY a corrected JSON response that fixes this specific error. "
        "Do not include markdown or commentary."
    )


RequestedFormat = Literal["text", "json_schema", "grammar"]


def placeholder_gbnf_grammar(schema: type[BaseModel]) -> str:
    """A minimal placeholder GBNF grammar - NOT a complete, schema-accurate
    grammar. Real JSON-schema-to-GBNF generation is a nontrivial ecosystem
    tool (llama.cpp ships one; see the theory doc's approach table) and is
    out of scope to reimplement here. This exists only so the pipeline's
    grammar-request code path (and FeatureNotSupported handling for
    runtimes without grammar support) is exercisable end to end for Lab 8's
    three-way comparison - it is not a production grammar.
    """
    field_names = ", ".join(schema.model_fields.keys())
    return f'root ::= "{{" ws "}}"  # placeholder grammar for fields: {field_names} - not schema-complete\nws ::= [ \\t\\n]*'


@dataclass(frozen=True)
class ExtractionResult(Generic[SchemaT]):
    fields: dict[str, Any]
    parsed: SchemaT | None
    confidence: ConfidenceLevel
    used_constrained_decoding: bool
    used_repair_retry: bool
    validation_error: str | None
    raw_output: str
    needs_review: bool


class ExtractionPipeline(Generic[SchemaT]):
    """Runs the full pipeline: prompt assembly -> constrained-decoding LLM
    call -> parse -> validate -> repair retry -> confidence scoring ->
    review-queue enqueue if needed. Persistence is the caller's job (theory
    doc: the pipeline stops at "persist result").
    """

    def __init__(
        self,
        runtime: Any,
        schema: type[SchemaT],
        *,
        required_fields: list[str] | None = None,
        max_repair_attempts: int = 1,
        review_queue: ReviewQueue | None = None,
        confidence_downgrade_to_review: ConfidenceLevel = "low",
        response_format_type: RequestedFormat = "json_schema",
    ) -> None:
        self.runtime = runtime
        self.schema = schema
        self.required_fields = required_fields if required_fields is not None else list(schema.model_fields.keys())
        self.max_repair_attempts = max_repair_attempts
        self.response_format_type = response_format_type
        # NOT `review_queue or ReviewQueue()`: ReviewQueue defines __len__,
        # so an empty (falsy) queue passed in would be silently replaced by
        # a fresh one, discarding the caller's queue instance.
        self.review_queue = review_queue if review_queue is not None else ReviewQueue()
        self.confidence_downgrade_to_review = confidence_downgrade_to_review

    def _build_primary_response_format(self) -> ResponseFormat:
        if self.response_format_type == "grammar":
            return ResponseFormat(type="grammar", grammar=placeholder_gbnf_grammar(self.schema))
        if self.response_format_type == "json_schema":
            return ResponseFormat(type="json_schema", schema=self.schema.model_json_schema())
        return ResponseFormat()  # "text": prompt-only mode, no structured decoding requested

    async def run(self, text: str, model: str) -> ExtractionResult[SchemaT]:
        prompt = build_extraction_prompt(text, self.schema)
        # "text" mode is deliberately prompt-only from the start, not a
        # fallback - only json_schema/grammar attempts count as constrained
        # decoding until/unless they hit FeatureNotSupported below.
        used_constrained_decoding = self.response_format_type != "text"

        request = LLMRequest(model=model, prompt=prompt, response_format=self._build_primary_response_format())
        try:
            response = await self.runtime.generate(request)
        except FeatureNotSupported:
            used_constrained_decoding = False
            request = LLMRequest(model=model, prompt=prompt)
            response = await self.runtime.generate(request)

        raw_output = response.text
        parsed_dict, parsed_model, validation_error = self._parse_and_validate(raw_output)

        used_repair_retry = False
        attempts = 0
        while parsed_model is None and attempts < self.max_repair_attempts:
            attempts += 1
            used_repair_retry = True
            repair_prompt = build_repair_prompt(prompt, raw_output, validation_error or "invalid JSON")
            repair_format = request.response_format if used_constrained_decoding else ResponseFormat()
            repair_request = LLMRequest(model=model, prompt=repair_prompt, response_format=repair_format)
            try:
                response = await self.runtime.generate(repair_request)
            except FeatureNotSupported:
                used_constrained_decoding = False
                repair_request = LLMRequest(model=model, prompt=repair_prompt)
                response = await self.runtime.generate(repair_request)
            raw_output = response.text
            parsed_dict, parsed_model, validation_error = self._parse_and_validate(raw_output)

        fields = parsed_dict if isinstance(parsed_dict, dict) else {}
        confidence = compute_confidence(
            ConfidenceInputs(
                extracted_fields=fields,
                required_fields=self.required_fields,
                used_repair_retry=used_repair_retry,
                used_constrained_decoding=used_constrained_decoding,
            )
        )
        needs_review = parsed_model is None or confidence == self.confidence_downgrade_to_review

        if needs_review:
            self.review_queue.enqueue(
                extracted_fields=fields,
                confidence=confidence,
                reason=validation_error or f"confidence={confidence}",
                source_text=text,
            )

        return ExtractionResult(
            fields=fields,
            parsed=parsed_model,
            confidence=confidence,
            used_constrained_decoding=used_constrained_decoding,
            used_repair_retry=used_repair_retry,
            validation_error=validation_error,
            raw_output=raw_output,
            needs_review=needs_review,
        )

    async def run_chunked(
        self, text: str, model: str, max_chars: int, overlap_chars: int = 0
    ) -> ExtractionResult[SchemaT]:
        """Lab 2: extract from long text using chunking, merging partial results."""
        chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        if len(chunks) <= 1:
            return await self.run(text, model)

        chunk_results = [await self.run(chunk, model) for chunk in chunks]
        merged = merge_partial_extractions([r.fields for r in chunk_results])

        used_constrained_decoding = all(r.used_constrained_decoding for r in chunk_results)
        used_repair_retry = any(r.used_repair_retry for r in chunk_results)
        confidence = compute_confidence(
            ConfidenceInputs(
                extracted_fields=merged.merged,
                required_fields=self.required_fields,
                used_repair_retry=used_repair_retry,
                used_constrained_decoding=used_constrained_decoding,
                had_conflicting_chunks=bool(merged.conflicting_fields),
            )
        )

        try:
            parsed_model = self.schema.model_validate(merged.merged)
            validation_error = None
        except ValidationError as exc:
            parsed_model = None
            validation_error = str(exc)

        needs_review = (
            parsed_model is None or confidence == self.confidence_downgrade_to_review or bool(merged.conflicting_fields)
        )
        if needs_review:
            reason = validation_error or (
                f"conflicting fields across chunks: {merged.conflicting_fields}"
                if merged.conflicting_fields
                else f"confidence={confidence}"
            )
            self.review_queue.enqueue(
                extracted_fields=merged.merged, confidence=confidence, reason=reason, source_text=text
            )

        return ExtractionResult(
            fields=merged.merged,
            parsed=parsed_model,
            confidence=confidence,
            used_constrained_decoding=used_constrained_decoding,
            used_repair_retry=used_repair_retry,
            validation_error=validation_error,
            raw_output="\n---\n".join(r.raw_output for r in chunk_results),
            needs_review=needs_review,
        )

    def _parse_and_validate(
        self, raw_output: str
    ) -> tuple[dict[str, Any] | None, SchemaT | None, str | None]:
        parsed = try_parse_json(raw_output)
        if parsed is None:
            return None, None, "output is not valid JSON"
        if not isinstance(parsed, dict):
            return None, None, f"expected a JSON object, got {type(parsed).__name__}"
        try:
            model = self.schema.model_validate(parsed)
        except ValidationError as exc:
            return parsed, None, str(exc)
        return parsed, model, None
