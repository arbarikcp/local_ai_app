"""Truncation strategies: drop-oldest and keep-system+last-N (theory doc
§3-4, 7, 11).

Every strategy operates on whole turn GROUPS (via group_turns()), never on
individual turns within a tool-call/tool-result pair - "tool-call and
tool-result turns must be kept as a unit" (Gotcha). Sticky turns/groups
(§7: system prompt, tools, current task, safety policy) are exempt from
both strategies.
"""

from __future__ import annotations

from .token_budget import ConversationBudget, TokenCounterFn, heuristic_token_counter, history_exceeds_budget
from .turn import Turn


def group_turns(turns: list[Turn]) -> list[list[Turn]]:
    """Group turns sharing the same non-None turn_group_id into one atomic
    unit; every other turn is its own singleton group. Preserves order.
    """
    groups: list[list[Turn]] = []
    group_index_by_id: dict[str, int] = {}
    for turn in turns:
        if turn.turn_group_id is not None and turn.turn_group_id in group_index_by_id:
            groups[group_index_by_id[turn.turn_group_id]].append(turn)
        else:
            groups.append([turn])
            if turn.turn_group_id is not None:
                group_index_by_id[turn.turn_group_id] = len(groups) - 1
    return groups


def group_is_sticky(group: list[Turn]) -> bool:
    return any(turn.sticky for turn in group)


def drop_oldest(
    turns: list[Turn], budget: ConversationBudget, token_counter: TokenCounterFn = heuristic_token_counter
) -> list[Turn]:
    """Remove the oldest non-sticky group at a time until under budget, or
    until nothing non-sticky is left to drop (even if still over budget -
    sticky content is never removed by this strategy).
    """
    groups = group_turns(turns)
    while True:
        flat = [t for g in groups for t in g]
        if not history_exceeds_budget(flat, budget, token_counter):
            return flat
        drop_index = next((i for i, g in enumerate(groups) if not group_is_sticky(g)), None)
        if drop_index is None:
            return flat
        groups.pop(drop_index)


def keep_system_plus_last_n(turns: list[Turn], n: int) -> list[Turn]:
    """Keep every sticky group plus the N most recent groups, in original order."""
    if n < 0:
        raise ValueError("n must be >= 0")
    groups = group_turns(turns)
    sticky_indices = {i for i, g in enumerate(groups) if group_is_sticky(g)}
    last_n_indices = set(range(max(0, len(groups) - n), len(groups)))
    keep_indices = sorted(sticky_indices | last_n_indices)
    return [t for i in keep_indices for t in groups[i]]
