"""Lab 1 — a chat loop with SQLite session persistence.
Lab 6 — /forget command (memory deletion).

The interactive REPL (run_repl) is a thin, untested wrapper; the testable
core is process_user_input(), which appends the user turn, calls the
runtime with the full history, appends and returns the assistant turn -
runtime-agnostic via Module 6's LLMRuntime interface (FakeRuntime in tests,
OllamaRuntime for real use).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.conversation.session_store import SessionStore  # noqa: E402
from local_ai_core.conversation.turn import Turn  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

FORGET_COMMAND = "/forget"


def is_forget_command(text: str) -> bool:
    return text.strip() == FORGET_COMMAND


def render_history(history: list[Turn]) -> str:
    """A simple, honest stand-in for a real chat-template renderer (Module
    6's MLXRuntime.render_prompt applies the model's own template) - good
    enough for this lab script, not a production prompt renderer.
    """
    return "\n".join(f"{t.role}: {t.content}" for t in history)


async def process_user_input(
    store: SessionStore,
    session_id: str,
    user_input: str,
    runtime,
    model: str,
    system_prompt: str | None = None,
) -> Turn:
    if not store.session_exists(session_id) and system_prompt:
        store.append_turn(session_id, Turn(role="system", content=system_prompt, sticky=True))

    store.append_turn(session_id, Turn(role="user", content=user_input))
    history = store.get_turns(session_id)

    response = await runtime.generate(LLMRequest(model=model, prompt=render_history(history)))

    assistant_turn = Turn(role="assistant", content=response.text)
    store.append_turn(session_id, assistant_turn)
    return assistant_turn


async def run_repl(store: SessionStore, session_id: str, runtime, model: str, system_prompt: str | None = None) -> None:
    """The actual interactive loop - untested by design (reads real stdin);
    all its logic lives in the tested process_user_input()/is_forget_command().
    """
    print(f"Chat session: {session_id} (type /forget to delete this session's memory, Ctrl-D to exit)")
    while True:
        try:
            user_input = input("> ")
        except EOFError:
            break
        if is_forget_command(user_input):
            store.delete_session(session_id)
            print("Session memory deleted.")
            continue
        assistant_turn = await process_user_input(store, session_id, user_input, runtime, model, system_prompt)
        print(assistant_turn.content)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import asyncio

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))
    from ollama_probe import is_ollama_available
    from local_ai_core.runtimes.ollama import OllamaRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--db-path", default="chat_sessions.db")
    parser.add_argument("--session-id", default="default-session")
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on the resourced 32GB Mac.",
            file=sys.stderr,
        )
        return 1

    async def _run():
        store = SessionStore(args.db_path)
        runtime = OllamaRuntime()
        try:
            await run_repl(store, args.session_id, runtime, args.model, system_prompt="You are a helpful assistant.")
        finally:
            await runtime.aclose()
            store.close()

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
