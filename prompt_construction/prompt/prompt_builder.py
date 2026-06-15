"""Unified prompt builder — renders Jinja2 templates with prepared context.

This module is the single entry point for prompt construction.
It receives already-prepared data and renders templates — no DB calls,
no LLM calls, no memory writes.

Typical usage::

    from prompt_construction.prompt.prompt_builder import build_prompt, PromptContext

    ctx = PromptContext(npc_name="Damon", ...)
    prompt = build_prompt(ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from prompt_construction.prompt.template_renderer import render


# ---------------------------------------------------------------------------
# PromptContext — all data needed to render the NPC prompt
# ---------------------------------------------------------------------------

@dataclass
class PromptContext:
    """All data needed to render the NPC prompt.

    Fields are grouped by template section.
    """

    # ── Identity ──────────────────────────────────────────────────────
    npc_name: str = ""
    character_type: str = "native"        # "original" | "native"

    # ── Persona (persona.jinja2) ──────────────────────────────────────
    persona_core: str = ""                # original: full persona; native: empty → default
    persona_background: Optional[str] = None
    persona_growth: Optional[str] = None
    speech_style: Optional[str] = None

    # ── Relationship (persona.jinja2) ─────────────────────────────────
    relationship_desc: str = ""
    interaction_style: str = ""

    # ── Memory (persona.jinja2 + memory.jinja2) ──────────────────────
    persona_memory: str = ""              # long-term persona text
    long_term_impression: str = ""        # summary_text
    relevant_past_events: str = ""        # event_text
    short_history: str = ""               # formatted current-turn history
    today_conversations: str = ""         # formatted mid-term memory

    # ── Environment (context.jinja2) ─────────────────────────────────
    weather: str = "晴朗"
    game_time: str = "未知"
    location: str = "未知"
    player_info: str = "healthy"
    today_actions: str = "无"
    attitude: str = "中立"

    # ── Dialogue (dialogue.jinja2) ───────────────────────────────────
    dialogue_examples: str = ""
    gift_rules: str = ""
    player_input: str = ""

    # ── Optional rules (dialogue.jinja2) ─────────────────────────────
    mood_rules: Optional[str] = None
    dialogue_constraints: Optional[str] = None
    do_list: List[str] = field(default_factory=list)
    dont_list: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template sections in order
# ---------------------------------------------------------------------------

_TEMPLATE_SECTIONS = [
    "npc/system.jinja2",
    "npc/persona.jinja2",
    "npc/memory.jinja2",
    "npc/context.jinja2",
    "npc/dialogue.jinja2",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(ctx: PromptContext) -> str:
    """Render the full NPC prompt from a PromptContext.

    Args:
        ctx: Fully populated PromptContext.

    Returns:
        Complete prompt string ready for the LLM.
    """
    context = {
        # Identity
        "npc_name":       ctx.npc_name,
        "character_type": ctx.character_type,

        # Persona
        "persona_core":       ctx.persona_core,
        "persona_background": ctx.persona_background,
        "persona_growth":     ctx.persona_growth,
        "speech_style":       ctx.speech_style,

        # Relationship
        "relationship_desc":  ctx.relationship_desc,
        "interaction_style":  ctx.interaction_style,

        # Memory
        "persona_text":          ctx.persona_memory,
        "summary_text":          ctx.long_term_impression,
        "event_text":            ctx.relevant_past_events,
        "formatted_history":     ctx.short_history,
        "today_conversations":   ctx.today_conversations,

        # Environment
        "weather":       ctx.weather,
        "game_time":     ctx.game_time,
        "location":      ctx.location,
        "player_info":   ctx.player_info,
        "today_actions": ctx.today_actions,
        "attitude":      ctx.attitude,

        # Dialogue
        "dialogue_examples": ctx.dialogue_examples,
        "gift_rules":        ctx.gift_rules,
        "player_input":      ctx.player_input,

        # Optional rules
        "mood_rules":           ctx.mood_rules,
        "dialogue_constraints": ctx.dialogue_constraints,
        "do_list":              ctx.do_list,
        "dont_list":            ctx.dont_list,
    }

    return "\n\n".join(render(section, **context) for section in _TEMPLATE_SECTIONS)
