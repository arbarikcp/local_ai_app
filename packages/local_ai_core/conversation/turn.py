"""Turn — the shared unit every conversation-management function in this
package operates on (theory doc §1). Every truncation/summarization
function works on whole Turns (or whole tool-call/result groups, §11),
never on substrings of a turn's content - "never hard-cut in the middle of
a user, assistant, or tool-result turn" (Conversation budget section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str
    tool_call_id: str | None = None
    turn_group_id: str | None = None
    sticky: bool = False
    """Exempt from every truncation strategy in this module (§7: system
    prompt, tools, current task, safety policy)."""


def to_chat_messages(turns: list[Turn]) -> list[dict[str, str]]:
    """Render Turns to the plain {"role": ..., "content": ...} message list
    shape most chat-template renderers (Module 6's MLXRuntime.render_prompt,
    Module 1's HFTokenizerCounter.count_chat) expect. Does NOT apply a chat
    template itself (§1: that's the adapter's job, not this package's).
    """
    return [{"role": t.role, "content": t.content} for t in turns]
