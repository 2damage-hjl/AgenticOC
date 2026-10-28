"""Deterministic dialogue key parser for Stardew Valley author keys.

Parses author keys like ``winter_Wed2``, ``rain``, ``GreenRain``,
``Resort``, ``breakUp``, ``divorced`` into structured script conditions
and any key-inferred required_flags.

This module is the single source of truth for key parsing logic.
The LLM must never decide hard conditions such as marriage, heart_min,
route, or required_flags.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASONS = {"spring", "summer", "fall", "winter"}
DAYS_OF_WEEK = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
VALID_HEART_VALUES = {2, 4, 6, 8, 10}

MARRIAGE_CHARACTERS = frozenset({
    "Abigail", "Alex", "Elliott", "Emily", "Haley", "Harvey",
    "Krobus", "Leah", "Maru", "Penny", "Sam", "Sebastian", "Shane",
})

# ---------------------------------------------------------------------------
# Default result template
# ---------------------------------------------------------------------------

def _default_result() -> dict[str, Any]:
    return {
        "season": ["any"],
        "day_of_week": ["any"],
        "day_of_month": ["any"],
        "heart_min": None,
        "relationship_gate": "any",
        "weather": ["any"],
        "special_key_type": "generic_dialogue",
        "variant_index": None,
        "parse_warnings": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dialogue_key(author_key: str, *, is_marriage: bool = False) -> dict[str, Any]:
    """Parse a dialogue author key into deterministic script conditions.

    Parameters
    ----------
    author_key:
        The raw key from the game JSON (e.g. ``winter_Wed2``, ``rain``).
    is_marriage:
        If True, the key comes from a MarriageDialogue file and may use
        marriage-specific patterns (Rainy_Day_0, Indoor_Day_0, etc.).

    Returns
    -------
    dict with keys:
        - script_conditions: parsed season / day / heart / weather / etc.
        - key_required_flags: list of required flags inferred from the key
          (e.g. ``["world.ginger_island.beach_resort.opened"]`` for Resort).
    """
    result = _default_result()
    key_required_flags: list[str] = []

    key = author_key.strip()
    lowered = key.lower()

    # ---- Exact-match special keys ----

    if lowered in {"divorced", "divorced_once"}:
        result["special_key_type"] = "divorced"
        result["relationship_gate"] = "divorced"
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    if lowered in {"breakup", "break_up", "breakup_dialogue"}:
        result["special_key_type"] = "breakup"
        result["relationship_gate"] = "breakup"
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    if lowered == "rain":
        result["weather"] = ["rainy"]
        result["special_key_type"] = "weather_rain"
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    if lowered == "greenrain" or lowered == "green_rain":
        result["weather"] = ["green_rain"]
        result["special_key_type"] = "weather_green_rain"
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    # ---- Resort keys ----

    if lowered.startswith("resort"):
        result["special_key_type"] = "ginger_island_resort"
        key_required_flags.append("world.ginger_island.beach_resort.opened")
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    # ---- Community Center (cc_) keys ----
    # cc_Begin, cc_Bridge, cc_Bus, cc_Minecart, cc_Boulder, cc_Complete
    # These are triggered by community center bundle completions → CC route only.

    if lowered.startswith("cc_"):
        result["special_key_type"] = "community_center_event"
        return {
            "script_conditions": result,
            "key_required_flags": key_required_flags,
            "key_route": "community_center",
        }

    # ---- Weather detection (partial) ----

    if "greenrain" in lowered or "green_rain" in lowered:
        result["weather"] = ["green_rain"]

    if "rain" in lowered:
        # "rain" appears in key but key is not exclusively "rain"
        if result["weather"] == ["any"]:
            result["weather"] = ["rainy"]

    # ---- Marriage-specific key patterns ----

    if is_marriage:
        _parse_marriage_key(key, result)
        # Marriage keys: trailing number is a variant index, NOT a heart level.
        # Married characters always have heart_min >= 10.
        result["heart_min"] = 10
        result["relationship_gate"] = "married"
        # Still try season/day extraction below for keys like spring_1

    # ---- Season + DayOfWeek + Heart pattern ----

    parts = key.split("_")

    # Season prefix: winter_Wed2, summer_Mon8
    if parts and parts[0].lower() in SEASONS:
        result["season"] = [parts[0].lower()]
        rest = "_".join(parts[1:])
    else:
        rest = key

    # DayOfWeek + optional heart: Wed2, Mon8, Sat10
    match = re.fullmatch(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)(2|4|6|8|10)?", rest)
    if match:
        result["day_of_week"] = [match.group(1)]
        if match.group(2) is not None:
            heart = int(match.group(2))
            result["heart_min"] = heart
            result["relationship_gate"] = f"heart_min_{heart}"
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    # Season + day-of-month: spring_13, summer_1
    if result["season"] != ["any"]:
        day_match = re.fullmatch(r"(\d{1,2})", rest)
        if day_match:
            result["day_of_month"] = [str(int(day_match.group(1)))]
            result["special_key_type"] = "date_specific_dialogue"
            return {"script_conditions": result, "key_required_flags": key_required_flags}

    # Standalone weekday: Wed, Mon
    if rest in DAYS_OF_WEEK:
        result["day_of_week"] = [rest]
        return {"script_conditions": result, "key_required_flags": key_required_flags}

    # Heart suffix detected but rest is not a clean pattern
    # Skip for marriage keys — trailing numbers are variant indices, not heart levels
    if not is_marriage:
        heart_suffix = re.search(r"(2|4|6|8|10)$", key)
        if heart_suffix:
            heart = int(heart_suffix.group(1))
            result["heart_min"] = heart
            result["relationship_gate"] = f"heart_min_{heart}"
            result["parse_warnings"].append(
                "Heart suffix detected, but full key pattern was not fully parsed."
            )

    # If nothing matched well, add a generic warning
    if result["special_key_type"] == "generic_dialogue" and not result["parse_warnings"]:
        if result["season"] == ["any"] and result["day_of_week"] == ["any"] and result["heart_min"] is None:
            result["parse_warnings"].append(
                f"Key '{author_key}' did not match any known pattern."
            )

    return {"script_conditions": result, "key_required_flags": key_required_flags}


# ---------------------------------------------------------------------------
# Marriage-specific helpers
# ---------------------------------------------------------------------------

def _parse_marriage_key(key: str, result: dict[str, Any]) -> None:
    """Parse marriage-specific key patterns (Rainy_Day_0, Indoor_Day_0, etc.).

    Trailing numbers in marriage keys are variant indices (e.g. Rainy_Day_0,
    Rainy_Day_1), NOT heart levels.  The caller is responsible for setting
    heart_min=10 and relationship_gate="married".
    """
    lowered = key.lower()

    # Extract trailing number as variant_index
    variant_match = re.search(r"_(\d+)$", key)
    if variant_match:
        result["variant_index"] = int(variant_match.group(1))

    # Rainy_Day / Rainy_Night
    if lowered.startswith("rainy_day") or lowered.startswith("rainy_night"):
        if result["weather"] == ["any"]:
            result["weather"] = ["rainy"]

    # Indoor_Day / Indoor_Night / Outdoor
    # (no weather change for these)

    # Good / Neutral / Bad mood
    if lowered.startswith("good_"):
        result["special_key_type"] = "marriage_mood_good"
    elif lowered.startswith("neutral_"):
        result["special_key_type"] = "marriage_mood_neutral"
    elif lowered.startswith("bad_"):
        result["special_key_type"] = "marriage_mood_bad"

    # OneKid / TwoKids
    if lowered.startswith("onekid"):
        result["special_key_type"] = "marriage_one_kid"
    elif lowered.startswith("twokids"):
        result["special_key_type"] = "marriage_two_kids"

    # spouseRoom / patio
    if lowered.startswith("spouseroom") or lowered.startswith("patio"):
        result["special_key_type"] = "marriage_spouse_room"

    # funLeave / funReturn
    if lowered.startswith("funleave") or lowered.startswith("funreturn"):
        result["special_key_type"] = "marriage_visit"


def extract_character_from_shared_marriage_key(key: str) -> tuple[str | None, str]:
    """Extract character name from a shared MarriageDialogue.json key.

    Shared keys look like ``Rainy_Day_Abigail`` or ``spring_Haley``.
    Returns ``(character_or_None, cleaned_author_key)``.
    If no character is found, returns ``(None, key)``.
    """
    parts = key.rsplit("_", 1)
    if len(parts) == 2:
        candidate = parts[1]
        if candidate in MARRIAGE_CHARACTERS:
            return candidate, parts[0]
    return None, key
