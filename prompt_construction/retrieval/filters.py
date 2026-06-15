"""Pure game-state filter functions — zero heavy dependencies.

This module contains all dialogue-record filtering logic that depends only
on Python stdlib.  No embedding model, no LanceDB, no sentence-transformers
required, making it safe for fast unit testing.

Filter pipeline (applied in order):
1. filter_by_context   — character / season / weather / heart / relationship
2. route_visible        — community-center vs Joja route gating
3. required_flags_visible — pipe-delimited flag checking
4. filter_by_lang_fallback — target_lang → en → any fallback
5. post_filter          — orchestrates 1-4
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Route visibility
# ---------------------------------------------------------------------------

ROUTE_GROUPS: dict[str, set[str]] = {
    "any": {"pre_choice", "community_center_completed", "joja_active", "joja_completed"},
    "pre_choice": {"pre_choice"},
    "non_joja": {"pre_choice", "community_center_completed"},
    "community_center": {"community_center_completed"},
    "non_cc": {"pre_choice", "joja_active", "joja_completed"},
    "joja": {"joja_active", "joja_completed"},
    "post_joja": {"joja_completed"},
}


def route_visible(record_route: str, game_route_state: str) -> bool:
    """Return True if a record with *record_route* is visible under *game_route_state*.

    - record_route="any" is always visible
    - Otherwise, the game_route_state must be in the allowed set for the record's route
    """
    if not record_route or record_route == "any":
        return True

    allowed = ROUTE_GROUPS.get(record_route, {record_route})
    return game_route_state in allowed


# ---------------------------------------------------------------------------
# Required flags visibility
# ---------------------------------------------------------------------------

def required_flags_visible(required_flags_str: str, game_flags: set[str]) -> bool:
    """Return True if all required flags are satisfied by game_flags.

    *required_flags_str* is a pipe-delimited string (e.g.
    ``"relationship.married_to.Abigail|world.ginger_island.beach_resort.opened"``).
    Empty string means no requirements → always visible.
    """
    if not required_flags_str:
        return True

    flags = [f.strip() for f in required_flags_str.split("|") if f.strip()]
    return all(f in game_flags for f in flags)


# ---------------------------------------------------------------------------
# Context field filtering (season, weather, day_of_week, etc.)
# ---------------------------------------------------------------------------

def _match_multi(field_value: str | None, target: str | None) -> bool:
    """Match a pipe/comma-delimited multi-value field against a target.

    Rules:
    - target is None → always match (no filter specified)
    - field_value is "any" or empty → always match (record accepts any)
    - Otherwise, field_value must contain target as a discrete token
    """
    if target is None:
        return True
    if not field_value or field_value == "any":
        return True
    # Normalise to comma-separated tokens
    tokens = [t.strip() for t in field_value.replace("|", ",").split(",") if t.strip()]
    return target in tokens


def _match_heart_min(heart_min_field: int | None, heart_level: int | None) -> bool:
    """Match heart_min threshold against current heart_level.

    Rules:
    - heart_level is None → always match (no filter)
    - heart_min_field is None or 0 → always match
    - Otherwise, heart_level must be >= heart_min_field
    """
    if heart_level is None:
        return True
    if heart_min_field is None:
        return True
    try:
        return heart_level >= int(heart_min_field)
    except (ValueError, TypeError):
        return True


def filter_by_context(
    records: list[dict],
    *,
    character: str | None = None,
    season: str | None = None,
    weather: str | None = None,
    day_of_week: str | None = None,
    day_of_month: str | None = None,
    heart_level: int | None = None,
    relationship: str | None = None,
) -> list[dict]:
    """Filter records by game-context fields.

    All parameters are optional; None means "do not filter on this field".

    Multi-value fields (season, weather, day_of_week, day_of_month) are
    matched inclusively: if the record's field contains the target value
    (or is "any"), it passes.

    heart_level is matched against heart_min: the player's heart level
    must be >= the record's heart_min threshold.

    relationship is matched against relationship_gate: must equal exactly,
    or the record's gate must be "any".
    """
    result = []
    for rec in records:
        # character: exact match (safety net for pre-search where clause)
        if character is not None:
            rec_char = rec.get("character", "")
            if rec_char != character:
                continue

        if not _match_multi(rec.get("season"), season):
            continue
        if not _match_multi(rec.get("weather"), weather):
            continue
        if not _match_multi(rec.get("day_of_week"), day_of_week):
            continue
        if not _match_multi(rec.get("day_of_month"), day_of_month):
            continue
        if not _match_heart_min(rec.get("heart_min"), heart_level):
            continue

        # relationship: exact match or "any"
        if relationship is not None:
            gate = rec.get("relationship_gate", "any") or "any"
            if gate != "any" and gate != relationship:
                continue

        result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Language filtering with fallback
# ---------------------------------------------------------------------------

DEFAULT_MIN_RESULTS = 3  # minimum before triggering lang fallback


def filter_by_lang_fallback(
    records: list[dict],
    target_lang: str | None,
    min_results: int = DEFAULT_MIN_RESULTS,
) -> list[dict]:
    """Filter by target_lang with fallback: target → en → any.

    Each returned record gets a ``fallback_reason`` field:
    - None (absent)  — matched target_lang directly
    - "fallback_en"  — target_lang had too few results, fell back to English
    - "fallback_any" — target_lang + en still too few, fell back to any language

    If target_lang is None, all records pass with no fallback_reason.

    Args:
        records: Pre-filtered records (context/route/flags already applied).
        target_lang: Preferred language code (e.g. "zh").
        min_results: Trigger fallback if target_lang results < this number.
    """
    if target_lang is None:
        return records

    # Step 1: target_lang results
    primary = [r for r in records if r.get("lang") == target_lang]

    if len(primary) >= min_results:
        return primary

    # Step 2: supplement with English
    seen_ids = {r.get("processed_id") for r in primary}
    en_fallback = []
    for r in records:
        if r.get("processed_id") in seen_ids:
            continue
        if r.get("lang") == "en":
            r_fb = {**r, "fallback_reason": "fallback_en"}
            en_fallback.append(r_fb)
            seen_ids.add(r.get("processed_id"))

    combined = primary + en_fallback
    if len(combined) >= min_results:
        return combined

    # Step 3: supplement with any remaining language
    any_fallback = []
    for r in records:
        if r.get("processed_id") in seen_ids:
            continue
        r_fb = {**r, "fallback_reason": "fallback_any"}
        any_fallback.append(r_fb)
        seen_ids.add(r.get("processed_id"))

    return combined + any_fallback


# ---------------------------------------------------------------------------
# Full post-filter pipeline
# ---------------------------------------------------------------------------

def post_filter(
    records: list[dict],
    game_flags: set[str],
    route_state: str,
    target_lang: str | None = None,
    min_lang_results: int = DEFAULT_MIN_RESULTS,
    *,
    character: str | None = None,
    season: str | None = None,
    weather: str | None = None,
    day_of_week: str | None = None,
    day_of_month: str | None = None,
    heart_level: int | None = None,
    relationship: str | None = None,
) -> list[dict]:
    """Apply context, route, required_flags filters, then language fallback.

    Language fallback is applied *last* so that min_lang_results counts
    only records that passed all other visibility checks.
    """
    filtered = filter_by_context(
        records,
        character=character,
        season=season,
        weather=weather,
        day_of_week=day_of_week,
        day_of_month=day_of_month,
        heart_level=heart_level,
        relationship=relationship,
    )

    visible = []
    for rec in filtered:
        rec_route = rec.get("route", "any") or "any"
        if not route_visible(rec_route, route_state):
            continue

        req_flags = rec.get("required_flags", "")
        if not required_flags_visible(req_flags, game_flags):
            continue

        visible.append(rec)

    # Language fallback applied after all other filters
    visible = filter_by_lang_fallback(visible, target_lang, min_lang_results)

    return visible
