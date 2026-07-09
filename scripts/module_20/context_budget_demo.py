"""Lab 2 - add a context budgeter. No new package code: Module 12's
`ContextBudget`/`pack_context()` (RAG retrieval context) and Module 8.5's
`ConversationBudget` (chat history) already do this for real - this script
composes both for one full request's token accounting, the way a
production app would actually spend its context window: system + tools +
conversation history + retrieved chunks + reserved answer space, all
against one shared `context_window`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.conversation.token_budget import ConversationBudget, heuristic_token_counter  # noqa: E402
from local_ai_core.conversation.turn import Turn  # noqa: E402
from local_ai_rag.context_packers.budget_packer import ContextBudget, pack_context  # noqa: E402
from local_ai_rag.embeddings.embedder import SearchResult  # noqa: E402

CONTEXT_WINDOW = 4096
RESERVED_SYSTEM = 200
RESERVED_TOOLS = 150
RESERVED_ANSWER = 512


def run_lab() -> dict:
    history = [
        Turn(role="user", content="How do I reset my password?"),
        Turn(role="assistant", content="Click 'Forgot password' on the sign-in page."),
        Turn(role="user", content="It says the reset link expired, what now?"),
    ]

    conversation_budget = ConversationBudget(
        context_window=CONTEXT_WINDOW,
        reserved_system=RESERVED_SYSTEM,
        reserved_tools=RESERVED_TOOLS,
        reserved_current_user_turn=30,
        reserved_answer=RESERVED_ANSWER,
    )
    history_tokens = heuristic_token_counter(history)

    # Whatever the conversation budget didn't use is available for retrieved
    # RAG context - the two budgeters share one context window, not two
    # independent ones.
    remaining_for_retrieval = max(0, conversation_budget.history_budget - history_tokens)

    candidates = [
        SearchResult(
            doc_id="password-reset-guide",
            score=0.92,
            text="Password reset links expire after 24 hours. Request a new one from the sign-in page.",
            metadata={"doc_id": "password-reset-guide"},
        ),
        SearchResult(
            doc_id="account-security-faq",
            score=0.71,
            text="For account security questions, see the security FAQ.",
            metadata={"doc_id": "account-security-faq"},
        ),
    ]
    retrieval_budget = ContextBudget(
        max_context_tokens=remaining_for_retrieval,
        reserved_for_system=0,
        reserved_for_question=0,
        reserved_for_answer=0,
    )
    packed = pack_context(candidates, retrieval_budget)

    return {
        "context_window": CONTEXT_WINDOW,
        "conversation_history_budget": conversation_budget.history_budget,
        "conversation_history_tokens_used": history_tokens,
        "remaining_for_retrieval": remaining_for_retrieval,
        "packed_chunk_count": len(packed),
        "packed_doc_ids": [c.doc_id for c in packed],
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 2 - context budgeter (composing Module 12 + Module 8.5 unchanged)\n\n"
        f"- Context window: {result['context_window']}\n"
        f"- Conversation history budget: {result['conversation_history_budget']} "
        f"(used {result['conversation_history_tokens_used']})\n"
        f"- Remaining for retrieval: {result['remaining_for_retrieval']}\n"
        f"- Packed {result['packed_chunk_count']} chunk(s): {result['packed_doc_ids']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
