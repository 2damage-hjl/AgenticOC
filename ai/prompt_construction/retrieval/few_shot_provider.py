"""Few-shot example provider — orchestrates retrieval, selection, formatting and fallback.

This is the single entry point for the few-shot pipeline.
It returns a structured result (not just a string) so downstream code can
log, debug, and make informed decisions.

Pipeline::

    player_input + game_state
        ↓  build rich query
    LanceDB search_dialogues
        ↓  metadata filter (防剧透 + 防关系错配)
    select_examples (去重、距离过滤、多样性)
        ↓  dynamic examples
    static examples from npc_config
        ↓  fallback / supplement
    format_examples → 风格参考文本
        ↓
    FewShotResult(text, source, count, debug)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from prompt_construction.retrieval.example_selector import (
    format_examples,
    select_examples,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified Example data structure
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """Internal standard format for a single few-shot example."""

    example_id: str = ""
    npc_id: str = ""
    content: str = ""
    tag: str = ""
    source: str = ""          # "lancedb" | "static"
    distance: float = 0.0     # LanceDB raw distance (lower = more similar)
    similarity: float = 0.0   # 1 − distance, for convenience
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FewShotResult — the return type of get_few_shot_examples()
# ---------------------------------------------------------------------------

@dataclass
class FewShotResult:
    """Structured return from the few-shot pipeline.

    Attributes:
        text:      Formatted few-shot text ready for prompt injection.
                   Empty string when source == "empty".
        source:    "dynamic" | "static" | "mixed" | "empty"
        count:     Number of examples in the final text.
        debug:     Dict with retrieval details (for logging, NOT for prompt).
        selected_records: The LanceDB record dicts that were selected (for trace).
    """

    text: str = ""
    source: str = "empty"     # dynamic | static | mixed | empty
    count: int = 0
    debug: dict[str, Any] = field(default_factory=dict)
    selected_records: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rich query construction
# ---------------------------------------------------------------------------

def build_rich_query(
    npc_id: str,
    player_input: str,
    relationship: str = "",
    location: str = "",
    weather: str = "",
    attitude: str = "",
) -> str:
    """Build a rich query string for LanceDB retrieval.

    Few-shot retrieval targets "角色语气和相似场景", NOT factual memory.
    So we include scene context but NOT long-term memory text.
    """
    parts = [
        f"NPC: {npc_id}",
        f"玩家输入: {player_input}",
    ]
    if relationship:
        parts.append(f"当前关系: {relationship}")
    if location:
        parts.append(f"当前地点: {location}")
    if weather:
        parts.append(f"天气: {weather}")
    if attitude:
        parts.append(f"当前态度: {attitude}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Relationship stage ordering (for gate filter)
# ---------------------------------------------------------------------------

# Numeric stage values — higher = deeper relationship.
# Used for comparison: gate must be <= current stage to pass.
_STAGE_VALUES: dict[str, int] = {
    "stranger":        0,
    "acquaintance":    2,
    "heart_min_2":     2,
    "heart_min_4":     4,
    "heart_min_6":     6,
    "heart_min_8":     8,
    "friend":          4,
    "close_friend":    6,
    "best_friend":     8,
    "dating":          8,
    "engaged":        10,
    "married":        12,
    "spouse":         12,
    "divorced":       10,
}


def _normalize_relationship_key(s: str) -> str:
    """Normalize a relationship string for lookup: lowercase, strip, replace spaces with underscores."""
    return s.lower().strip().replace(" ", "_")


def _relationship_to_stage_value(relationship: str) -> int:
    """Map a relationship string to a numeric stage value.

    Returns -1 for unknown / empty (means "don't filter").
    """
    if not relationship:
        return -1
    key = _normalize_relationship_key(relationship)
    if key in _STAGE_VALUES:
        return _STAGE_VALUES[key]
    # Fallback: try substring matching for compound keys
    # e.g. "married_to_Abigail" → "married"
    for stage_name, value in _STAGE_VALUES.items():
        if key.startswith(stage_name):
            return value
    return -1


def _gate_to_stage_value(gate: str) -> int:
    """Map a relationship_gate string (from LanceDB record) to a numeric stage value.

    Examples: "heart_min_4" → 4, "married" → 12, "any" → -1 (no restriction)
    Returns -1 for "any" / empty (no restriction).
    """
    if not gate or gate == "any":
        return -1  # no restriction
    key = _normalize_relationship_key(gate)
    # Handle "married_to_NPC" pattern
    if key.startswith("married_to_"):
        key = "married"
    if key in _STAGE_VALUES:
        return _STAGE_VALUES[key]
    return -1


def filter_by_relationship_gate(
    records: list[dict],
    current_relationship: str,
) -> list[dict]:
    """Remove records whose relationship_gate exceeds current relationship stage.

    This prevents spoilers — e.g. a married dialogue appearing when the player
    is only at "friend" stage.

    Rules:
    - If current_relationship is unknown → don't filter (safe default)
    - If gate is "any" or empty → always pass
    - If gate's stage value > current stage value → blocked (spoiler risk)
    """
    current_val = _relationship_to_stage_value(current_relationship)
    if current_val < 0:
        # Unknown relationship → don't filter (safe default)
        return records

    result = []
    for rec in records:
        gate = rec.get("relationship_gate", "any") or "any"
        gate_val = _gate_to_stage_value(gate)
        if gate_val < 0:
            # Gate is "any" or unknown → always allow
            result.append(rec)
            continue
        if gate_val <= current_val:
            result.append(rec)
        # else: gate requires a higher relationship stage → skip (防剧透)

    return result


# ---------------------------------------------------------------------------
# Core: get_few_shot_examples()
# ---------------------------------------------------------------------------

# Thresholds
MIN_DYNAMIC_FOR_PURE = 3       # ≥3 dynamic → source = "dynamic"
MAX_DISTANCE = 0.75            # discard candidates with distance > this (None = no filter)
FINAL_K = 3                    # target number of final examples


def get_few_shot_examples(
    npc_id: str,
    state: dict,
    npc_config: dict,
) -> FewShotResult:
    """Orchestrate the few-shot pipeline with layered fallback.

    Returns a FewShotResult with:
    - text:      formatted examples (or "" for empty)
    - source:    "dynamic" | "static" | "mixed" | "empty"
    - count:     number of examples
    - debug:     {npc_id, source, count, retrieved_ids, distances, distance_distribution, fallback_reason}

    For ``character_type == "original"`` NPCs (user-created, like Damon),
    LanceDB data is typically low-quality scene triggers rather than genuine
    dialogue style examples.  These NPCs should rely on their hand-crafted
    static_examples from the NPC JSON config, not on dynamically retrieved
    LanceDB records.
    """
    player_input = state.get("last_user_input", "")
    relationship = state.get("relationship", "")
    static_text = npc_config.get("static_examples", "")
    character_type = npc_config.get("character_type", "native")

    # Build current_state dict for select_examples (scene-type filter + closeness)
    current_state = {
        "relationship": relationship,
        "is_birthday": state.get("is_birthday", False),
        "is_festival": state.get("is_festival", False),
        "is_gifting": state.get("is_gifting", False),
    }

    # ── Build rich query ──────────────────────────────────────────
    rich_query = build_rich_query(
        npc_id=npc_id,
        player_input=player_input,
        relationship=relationship,
        location=state.get("location", ""),
        weather=state.get("weather", ""),
        attitude=state.get("attitude", ""),
    )

    debug: dict[str, Any] = {
        "npc_id": npc_id,
        "character_type": character_type,
        "rich_query": rich_query[:200],  # truncate for log safety
        "retrieved_ids": [],
        "distances": [],
        "distance_distribution": {},
        "fallback_reason": None,
    }

    # ── Original NPC shortcut: skip LanceDB, use static only ──────
    # Original NPCs have low-quality LanceDB data (scene triggers, not
    # dialogue style), so dynamic retrieval hurts more than it helps.
    if character_type == "original":
        has_static = bool(static_text and static_text != "No dialogue examples.")
        if has_static:
            result = FewShotResult(
                text=static_text,
                source="static",
                count=0,
                debug={**debug, "fallback_reason": "original_npc_skipped_lancedb"},
            )
            logger.info(
                f"[few-shot] npc={npc_id} source=static count=0 "
                f"dynamic=0 static=True reason=original_npc_skipped_lancedb"
            )
            return result
        else:
            result = FewShotResult(
                text="",
                source="empty",
                count=0,
                debug={**debug, "fallback_reason": "original_npc_no_static_no_lancedb"},
            )
            logger.info(
                f"[few-shot] npc={npc_id} source=empty count=0 "
                f"dynamic=0 static=False reason=original_npc_no_static_no_lancedb"
            )
            return result

    # ── Try LanceDB retrieval ─────────────────────────────────────
    dynamic_examples: list[dict] = []
    try:
        from prompt_construction.retrieval.query_lancedb import search_dialogues, get_embedding_model

        # Ensure model is loaded (first call caches it; subsequent calls are no-op)
        try:
            _emb_model = get_embedding_model()
            debug["model_loaded"] = True
        except Exception as emb_err:
            debug["model_loaded"] = False
            debug["fallback_reason"] = f"embedding_model_failed: {emb_err}"
            logger.error(f"Embedding model load failed for {npc_id}: {emb_err}")
            raise  # will be caught by outer except → fallback

        # Derive game flags from state + relationship
        game_flags: set[str] = set(state.get("game_flags", []))
        if relationship and relationship.lower() in ("spouse", "married"):
            game_flags.add(f"relationship.married_to.{npc_id}")

        # Derive heart_level from relationship string
        heart_level = _derive_heart_level(relationship)

        raw_results = search_dialogues(
            query_text=rich_query,
            target_lang="zh",
            game_flags=game_flags,
            route_state=state.get("route", "community_center_completed"),
            character=npc_id,
            season=state.get("season", "").lower() or None,
            weather=state.get("weather", "").lower() or None,
            heart_level=heart_level,
            # NOTE: do NOT pass relationship to search_dialogues — it does
            # exact-match filtering on relationship_gate which rejects
            # heart_min_N gates.  The anti-spoiler filter is applied below
            # via filter_by_relationship_gate() instead.
            relationship=None,
            top_k=10,
        )

        # Filter by relationship gate (防剧透)
        filtered = filter_by_relationship_gate(raw_results, relationship)

        # Select with max_distance + current_state for scene-type filter & closeness
        selected = select_examples(
            filtered,
            target_lang="zh",
            min_output=MIN_DYNAMIC_FOR_PURE,
            max_output=FINAL_K + 2,  # allow some buffer for mixed mode
            max_distance=MAX_DISTANCE,
            current_state=current_state,
        )

        dynamic_examples = selected
        debug["retrieved_ids"] = [
            r.get("processed_id", "") for r in dynamic_examples
        ]
        debug["distances"] = [
            round(float(r.get("_distance", 0.5)), 4) for r in dynamic_examples
        ]

        # Record distance distribution for monitoring
        all_dists = sorted(float(r.get("_distance", 0.5)) for r in filtered)
        if all_dists:
            p50_idx = len(all_dists) // 2
            p95_idx = int(len(all_dists) * 0.95)
            debug["distance_distribution"] = {
                "min": round(all_dists[0], 4),
                "p50": round(all_dists[min(p50_idx, len(all_dists) - 1)], 4),
                "p95": round(all_dists[min(p95_idx, len(all_dists) - 1)], 4),
                "max": round(all_dists[-1], 4),
                "count_before_select": len(all_dists),
            }

    except Exception as e:
        debug["fallback_reason"] = f"lancedb_error: {e}"
        logger.warning(f"LanceDB retrieval failed for {npc_id}: {e}")

    # ── Layered fallback ──────────────────────────────────────────
    dynamic_count = len(dynamic_examples)
    has_static = bool(static_text and static_text != "No dialogue examples.")

    if dynamic_count >= MIN_DYNAMIC_FOR_PURE:
        # Pure dynamic
        text = format_examples(dynamic_examples)
        source = "dynamic"
        count = dynamic_count
    elif dynamic_count > 0 and has_static:
        # Mixed: dynamic + static supplement
        dynamic_text = format_examples(dynamic_examples)
        text = dynamic_text + "\n" + static_text
        source = "mixed"
        count = dynamic_count  # only count dynamic ones for debug
        debug["fallback_reason"] = "insufficient_dynamic_supplemented_by_static"
    elif dynamic_count > 0 and not has_static:
        # Only a few dynamic, no static available
        text = format_examples(dynamic_examples)
        source = "dynamic"
        count = dynamic_count
        debug["fallback_reason"] = "insufficient_dynamic_no_static_available"
    elif has_static:
        # Pure static
        text = static_text
        source = "static"
        count = 0  # static examples are pre-formatted, no individual count
        debug["fallback_reason"] = "lancedb_returned_zero"
    else:
        # Empty — nothing at all
        text = ""
        source = "empty"
        count = 0
        debug["fallback_reason"] = "no_dynamic_no_static"

    # ── Debug log ─────────────────────────────────────────────────
    result = FewShotResult(
        text=text,
        source=source,
        count=count,
        debug=debug,
        selected_records=dynamic_examples,
    )

    logger.info(
        f"[few-shot] npc={npc_id} source={source} count={count} "
        f"dynamic={dynamic_count} static={has_static} "
        f"reason={debug.get('fallback_reason', 'none')}"
    )

    return result


# ---------------------------------------------------------------------------
# Heart level derivation from relationship string
# ---------------------------------------------------------------------------

def _derive_heart_level(relationship: str) -> int | None:
    """Derive heart_level from relationship string for LanceDB filter.

    Mapping:
    - stranger → 0, acquaintance → 2, friend → 4,
    - close friend → 6, best friend → 8, dating → 8,
    - engaged → 10, spouse/married → 10+, divorced → 10
    """
    if not relationship:
        return None
    rl = relationship.lower().strip()
    mapping = {
        "stranger": 0,
        "acquaintance": 2,
        "friend": 4,
        "close friend": 6,
        "best friend": 8,
        "dating": 8,
        "engaged": 10,
        "spouse": 12,
        "married": 12,
        "divorced": 10,
    }
    return mapping.get(rl)


# ---------------------------------------------------------------------------
# Public convenience: get_few_shot_text()
# ---------------------------------------------------------------------------

def get_few_shot_text(npc_id: str, state: dict, npc_config: dict) -> str:
    """Convenience wrapper that returns just the text string.

    Use get_few_shot_examples() when you need source/debug info.
    """
    result = get_few_shot_examples(npc_id, state, npc_config)
    return result.text