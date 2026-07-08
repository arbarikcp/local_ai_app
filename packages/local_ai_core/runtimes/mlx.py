"""MLXRuntime — LLMRuntime adapter for MLX / mlx-lm.

mlx_lm has no built-in server (Module 5 theory doc §2) - this adapter calls
mlx_lm's load/generate/stream_generate functions in-process. Those
functions are synchronous, so every call here runs via asyncio.to_thread
rather than directly on the event loop (theory doc Gotchas: "blocking calls
inside an async server can serialize requests").

load/generate/stream_generate are injected via constructor - the same
dependency-injection principle as every adapter in this module - so tests
substitute fakes without needing MLX installed or Apple Silicon (this
repo's machine constraint: no model runtime installed here at all).
"""

from __future__ import annotations

import asyncio
import queue as queue_module
import threading
from typing import Any, AsyncIterator, Callable

from .base import MetricsHook, NullMetricsHook, Timer, ensure_trace_id
from .errors import FeatureNotSupported, InvalidModelResponse, ModelNotLoaded
from .types import LLMRequest, LLMResponse

LoadFn = Callable[[str], tuple[Any, Any]]
GenerateFn = Callable[..., str]
StreamGenerateFn = Callable[..., Any]


def _real_load(model_id: str) -> tuple[Any, Any]:
    from mlx_lm import load

    return load(model_id)


def _real_generate(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate

    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def _real_stream_generate(model: Any, tokenizer: Any, *, prompt: str, max_tokens: int) -> Any:
    from mlx_lm import stream_generate

    return stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def render_prompt(tokenizer: Any, request: LLMRequest) -> str:
    """Pure-ish translation: LLMRequest -> the single prompt string mlx_lm
    expects, using the tokenizer's own chat template when available (Module
    1 §5's rule: use the model's own chat template, never hand-roll one).
    Falls back to plain concatenation only if the tokenizer has no chat
    template - rare, but tokenizer-dependent.
    """
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if request.system:
        return f"{request.system}\n\n{request.prompt}"
    return request.prompt


def _reject_unsupported_response_format(request: LLMRequest) -> None:
    if request.response_format.type in ("json_schema", "grammar"):
        raise FeatureNotSupported(
            f"MLX/mlx-lm has no built-in constrained decoding for "
            f"response_format.type={request.response_format.type!r} "
            "(see scripts/module_05/feature_matrix.py)"
        )


class MLXRuntime:
    def __init__(
        self,
        *,
        load_fn: LoadFn = _real_load,
        generate_fn: GenerateFn = _real_generate,
        stream_generate_fn: StreamGenerateFn = _real_stream_generate,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        self._load_fn = load_fn
        self._generate_fn = generate_fn
        self._stream_generate_fn = stream_generate_fn
        self.metrics_hook: MetricsHook = metrics_hook or NullMetricsHook()
        # Loaded models stay resident for this runtime's lifetime, matching
        # llama.cpp-style behavior rather than Ollama's keep_alive eviction
        # (Module 5 theory doc §5) - MLX has no built-in unload mechanism.
        self._loaded_models: dict[str, tuple[Any, Any]] = {}

    async def _get_or_load(self, model_id: str) -> tuple[Any, Any]:
        if model_id not in self._loaded_models:
            try:
                self._loaded_models[model_id] = await asyncio.to_thread(self._load_fn, model_id)
            except Exception as exc:
                raise ModelNotLoaded(f"Could not load MLX model '{model_id}': {exc}", cause=exc) from exc
        return self._loaded_models[model_id]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        request = ensure_trace_id(request)
        _reject_unsupported_response_format(request)

        timer = Timer()
        model, tokenizer = await self._get_or_load(request.model)
        rendered_prompt = render_prompt(tokenizer, request)
        try:
            text = await asyncio.to_thread(
                self._generate_fn, model, tokenizer, prompt=rendered_prompt, max_tokens=request.max_tokens
            )
        except Exception as exc:
            error = InvalidModelResponse(f"MLX generation failed: {exc}", cause=exc)
            self.metrics_hook.on_request(request, None, error, timer.elapsed_ms)
            raise error from exc

        prompt_tokens, completion_tokens = _safe_token_counts(tokenizer, rendered_prompt, text)
        response = LLMResponse(
            text=text,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=timer.elapsed_ms,
            stop_reason="stop",
        )
        self.metrics_hook.on_request(request, response, None, timer.elapsed_ms)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        request = ensure_trace_id(request)
        _reject_unsupported_response_format(request)

        model, tokenizer = await self._get_or_load(request.model)
        rendered_prompt = render_prompt(tokenizer, request)

        # mlx_lm.stream_generate is a synchronous generator. Bridge it to
        # this async generator via a background thread and a thread-safe
        # queue, so the event loop is never blocked waiting for the next
        # token (theory doc Gotchas) - this is genuine incremental
        # streaming, not "run it all, then yield," which would defeat the
        # point of measuring streaming TTFT (Module 5 theory doc §3).
        q: queue_module.Queue = queue_module.Queue()
        sentinel = object()

        def _produce() -> None:
            try:
                for item in self._stream_generate_fn(
                    model, tokenizer, prompt=rendered_prompt, max_tokens=request.max_tokens
                ):
                    q.put(item.text)
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer below
                q.put(exc)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()
        try:
            while True:
                item = await asyncio.to_thread(q.get)
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise InvalidModelResponse(f"MLX streaming failed: {item}", cause=item) from item
                yield item
        finally:
            thread.join(timeout=5)

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        _, tokenizer = await self._get_or_load(model)
        try:
            return await asyncio.to_thread(tokenizer.encode, rendered_prompt)
        except Exception as exc:
            raise InvalidModelResponse(f"MLX tokenization failed: {exc}", cause=exc) from exc


def _safe_token_counts(tokenizer: Any, prompt: str, completion: str) -> tuple[int | None, int | None]:
    """Best-effort real token counts from the tokenizer; None (not a fake
    estimate) if the tokenizer doesn't support .encode (Module 1 §5's rule:
    never estimate when a real count is unavailable).
    """
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None, None
    try:
        return len(encode(prompt)), len(encode(completion))
    except Exception:  # noqa: BLE001 - token counting is best-effort metadata
        return None, None
