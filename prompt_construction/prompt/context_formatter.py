"""Format memory and context data into prompt-ready strings.

This module is responsible only for turning raw data structures
into human-readable strings.  No DB calls, no LLM calls, no side effects.

Usage in graph.py::

    short_history = format_short_history(recent_messages)
    mid_memory    = format_mid_memory(mid_contents)
    long_memory   = format_long_memory(ctx)
    actions_text  = format_today_actions(today_actions)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Short-term conversation history
# ---------------------------------------------------------------------------

def format_short_history(recent_messages: list[dict]) -> str:
    """Format short-term conversation history into a string.

    Args:
        recent_messages: List of dicts with ``role`` (``"player"`` | ``"npc"``)
            and ``content`` keys.

    Returns:
        Formatted string like ``"玩家: xxx\\nnpc: yyy"``, or empty string.
    """
    if not recent_messages:
        return ""
    lines: list[str] = []
    for m in recent_messages:
        role = "玩家" if m.get("role") == "player" else "npc"
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mid-term memory (today's conversations)
# ---------------------------------------------------------------------------

def format_mid_memory(mid_memories: list[dict]) -> str:
    """Format mid-term memory (today's conversations) into a string.

    Args:
        mid_memories: List of dicts with ``location`` and ``content`` keys.

    Returns:
        Formatted string like ``"- [地点:xxx] content"``, or ``"暂无"``.
    """
    if not mid_memories:
        return "暂无"
    lines: list[str] = []
    for mem in mid_memories:
        location = mem.get("location", "未知")
        content = mem.get("content", "")
        lines.append(f"- [地点:{location}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

def format_long_memory(ctx: dict | None) -> dict[str, str]:
    """Extract and format long-term memory from context manager output.

    Args:
        ctx: Dict with ``persona_text``, ``summary_text``, ``event_text`` keys,
            or ``None``.

    Returns:
        Dict with keys ``persona_text``, ``summary_text``, ``event_text``.
    """
    if not ctx:
        return {"persona_text": "", "summary_text": "", "event_text": ""}
    return {
        "persona_text": ctx.get("persona_text", ""),
        "summary_text": ctx.get("summary_text", ""),
        "event_text":  ctx.get("event_text", ""),
    }


# ---------------------------------------------------------------------------
# Today's player actions
# ---------------------------------------------------------------------------

def format_today_actions(today_actions: list[str]) -> str:
    """Format today's player actions into a string.

    Args:
        today_actions: List of action description strings.

    Returns:
        Chinese-comma-separated string, or ``"无"``.
    """
    if not today_actions:
        return "无"
    return "、".join(str(a) for a in today_actions)
