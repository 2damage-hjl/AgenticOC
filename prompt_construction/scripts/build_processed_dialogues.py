"""Build processed dialogue records from canonical files.

Reads canonical records from ``data/canonical/{Character}_canonical.json``,
expands each multilingual record into per-language processed records, applies
text normalization, generates hashes and retrieval text, and writes results to
``data/processed/{Character}_processed.json``.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup – ensure prompt_construction package is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PKG_DIR = Path(__file__).resolve().parent.parent

from prompt_construction.utils.json_io import load_json, save_json
from prompt_construction.utils.text_normalize import normalize_dialogue_text

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
CANONICAL_DIR = _PKG_DIR / "data" / "canonical"
PROCESSED_DIR = _PKG_DIR / "data" / "processed"
CONFIG_PATH = _PKG_DIR / "configs" / "index_config.yaml"


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def _normalize_raw(text: str) -> str:
    """Basic normalization preserving Stardew tokens.

    - trim whitespace
    - CRLF → LF
    - compress repeated spaces/tabs
    """
    return normalize_dialogue_text(text)


def _normalize_display(text: str) -> str:
    """Normalize for prompt display.

    Applies raw normalization, then:
    - strip $query conditionals (PLAYER_NPC_RELATIONSHIP, current any married roommate, etc.)
    - strip $d route conditionals (should already be split at canonical level)
    - replace #$b# (line break token) with newline
    - replace @ with {player_name}
    - replace %spouse with {spouse_name}
    - strip #$e# (dialogue continuation marker)
    - remove portrait/mood tokens like $s, $h, $u, $k, etc.
    - split on | (multi-response separator), keep first meaningful part
    """
    t = _normalize_raw(text)
    # Strip $query conditionals: $query PLAYER_NPC_RELATIONSHIP ...#
    t = re.sub(r"\$query\s+[^#]*#", "", t)
    # Strip any remaining $d conditionals (safety net)
    t = re.sub(r"\$d\s+(joja|cc)\s*#.*?\|", "", t, flags=re.IGNORECASE)
    t = re.sub(r"#\$b#", "\n", t)
    # Strip #$e# (end-of-response marker)
    t = re.sub(r"#\$e#", "", t)
    t = t.replace("@", "{player_name}")
    t = t.replace("%spouse", "{spouse_name}")
    # Strip Stardew portrait/mood tokens
    t = re.sub(r"\$[a-zA-Z]+", "", t)
    # Split on | — multi-response separator; keep first meaningful part
    # Some texts have "Response1|Response2" for multiple dialogue boxes
    parts = t.split("|")
    t = parts[0].strip()
    # Clean up extra whitespace left after token removal
    t = re.sub(r"[ \t]+", " ", t).strip()
    # Clean up leading/trailing newlines
    t = t.strip("\n")
    return t


def _normalize_embedding(text: str) -> str:
    """Normalize for embedding.

    Applies raw normalization, then:
    - strip $query conditionals (PLAYER_NPC_RELATIONSHIP, current any married roommate, etc.)
    - strip $d route conditionals (should already be split at canonical level)
    - replace #$b# with space
    - replace @ with "player"
    - replace %spouse with "spouse"
    - strip #$e# (dialogue continuation marker)
    - strip remaining Stardew control tokens like $h, $s, $u, $k, etc.
    - split on | (multi-response separator), keep first meaningful part
    """
    t = _normalize_raw(text)
    # Strip $query conditionals: $query PLAYER_NPC_RELATIONSHIP ...#
    t = re.sub(r"\$query\s+[^#]*#", "", t)
    # Strip any remaining $d conditionals (safety net)
    t = re.sub(r"\$d\s+(joja|cc)\s*#.*?\|", "", t, flags=re.IGNORECASE)
    t = re.sub(r"#\$b#", " ", t)
    # Strip #$e# (end-of-response marker)
    t = re.sub(r"#\$e#", "", t)
    t = t.replace("@", "player")
    t = t.replace("%spouse", "spouse")
    # Strip Stardew portrait/mood tokens: $h, $s, $u, $k, $l, $blush, etc.
    t = re.sub(r"\$[a-zA-Z]+", "", t)
    # Split on | — multi-response separator; keep first meaningful part
    parts = t.split("|")
    t = parts[0].strip()
    # Clean up extra whitespace left after token removal
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Scene type inference
# ---------------------------------------------------------------------------

# Patterns that map author_key → scene_type
_SCENE_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:Accept|Reject)BirthdayGift", re.IGNORECASE), "birthday"),
    (re.compile(r"birthday", re.IGNORECASE), "birthday"),
    (re.compile(r"^rain$|^greenrain$|^green_rain$", re.IGNORECASE), "weather"),
    (re.compile(r"rainy_day|rainy_night", re.IGNORECASE), "weather"),
    (re.compile(r"^resort", re.IGNORECASE), "location"),
    (re.compile(r"^cc_", re.IGNORECASE), "special"),
    (re.compile(r"divorced|breakup|break_up", re.IGNORECASE), "special"),
    (re.compile(r"^(good|neutral|bad)_", re.IGNORECASE), "marriage"),
    (re.compile(r"^(onekid|twokids)", re.IGNORECASE), "marriage"),
    (re.compile(r"^(spouseroom|patio)", re.IGNORECASE), "marriage"),
    (re.compile(r"^(funleave|funreturn)", re.IGNORECASE), "marriage"),
    (re.compile(r"^(Indoor|Outdoor|Rainy)_(Day|Night)", re.IGNORECASE), "marriage"),
    (re.compile(r"^GreenRain", re.IGNORECASE), "weather"),
    (re.compile(r"AcceptGift|RejectGift", re.IGNORECASE), "gift"),
    (re.compile(r"AcceptBouquet|RejectBouquet", re.IGNORECASE), "special"),
    (re.compile(r"AcceptPendant|RejectPendant", re.IGNORECASE), "special"),
    (re.compile(r"dumped", re.IGNORECASE), "special"),
    (re.compile(r"WipedMemory", re.IGNORECASE), "special"),
    # General dialogue with author_key "married" — NPCs commenting on player's marriage
    (re.compile(r"^married$", re.IGNORECASE), "marriage"),
    # Festival keys — Stardew uses festival-specific keys like eggFestival, flowerFestival, etc.
    (re.compile(r"(?:egg|flower|luau|stardew|fair|festival|ice|feast|night|carnival|mermaid)", re.IGNORECASE), "festival"),
    # Season + day-of-month patterns like spring_13 can be festival dates
    # These are handled below in _infer_scene_type
]

# Known festival day-of-month numbers per season
_FESTIVAL_DAYS: dict[str, set[int]] = {
    "spring": {13, 24},       # Egg Festival (13), Flower Dance (24)
    "summer": {11, 28},       # Luau (11), Dance of the Moonlight Jellies (28)
    "fall":   {16, 27},       # Stardew Valley Fair (16), Spirit's Eve (27)
    "winter": {8, 25},        # Festival of Ice (8), Feast of the Winter Star (25),
                                  # Night Market (15-17 handled separately)
}


def _infer_scene_type(
    author_key: str,
    dialogue_type: str,
    script_conditions: dict[str, Any],
) -> str:
    """Infer scene_type from author_key, dialogue_type, and script_conditions.

    Priority: birthday > festival > gift > marriage > special > weather > location > seasonal > daily

    Returns one of: daily, seasonal, weather, location, birthday, festival,
                     marriage, special, gift
    """
    key = author_key.strip()

    # Marriage dialogue → always "marriage"
    if dialogue_type == "marriage_dialogue":
        return "marriage"

    # Check known patterns
    for pattern, scene_type in _SCENE_TYPE_PATTERNS:
        if pattern.search(key):
            return scene_type

    # Check special_key_type from script_conditions
    skt = script_conditions.get("special_key_type", "")
    if skt == "weather_rain" or skt == "weather_green_rain":
        return "weather"
    if skt == "community_center_event":
        return "special"
    if skt == "ginger_island_resort":
        return "location"
    if skt in ("breakup", "divorced"):
        return "special"
    if skt == "date_specific_dialogue":
        # Check if the date is a known festival day
        seasons = script_conditions.get("season", [])
        day_months = script_conditions.get("day_of_month", [])
        for season in (seasons if isinstance(seasons, list) else [seasons]):
            if season in _FESTIVAL_DAYS:
                for dm in (day_months if isinstance(day_months, list) else [day_months]):
                    try:
                        if int(dm) in _FESTIVAL_DAYS[season]:
                            return "festival"
                    except (ValueError, TypeError):
                        pass
        return "seasonal"

    # Season-specific but generic key → seasonal
    seasons = script_conditions.get("season", [])
    if isinstance(seasons, list) and len(seasons) == 1 and seasons[0] != "any":
        return "seasonal"

    # Weather-specific
    weathers = script_conditions.get("weather", [])
    if isinstance(weathers, list) and any(w != "any" for w in weathers):
        return "weather"

    return "daily"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(record: dict[str, Any]) -> dict[str, Any]:
    """Build context dict from canonical record fields."""
    sc = record.get("script_conditions", {})

    def _arr(val: Any) -> list:
        if val is None:
            return ["any"]
        if isinstance(val, list):
            return val if val else ["any"]
        return [val]

    def _scalar(val: Any, default: str = "any") -> str:
        if val is None:
            return default
        return str(val)

    def _num(val: Any) -> int | None:
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    # source_key is the original author_key, not a file path
    source_key = record.get("author_key", "")

    ctx: dict[str, Any] = {
        "source_key": source_key,
        "key_type": record.get("dialogue_type", "any"),
        "special_key_type": _scalar(sc.get("special_key_type")),
        "season": _arr(sc.get("season")),
        "day_of_week": _arr(sc.get("day_of_week")),
        "day_of_month": _arr(sc.get("day_of_month")),
        "weather": _arr(sc.get("weather")),
        "time_period": _arr(sc.get("time_period", ["any"])),
        "scene": _scalar(sc.get("scene")) if "scene" in sc else "any",
    }

    # Marriage: heart_min is null (variant indices are NOT heart thresholds),
    # relationship_gate is "married_to_{character}"
    is_marriage = record.get("dialogue_type") == "marriage_dialogue"
    if is_marriage:
        ctx["heart_min"] = None
        ctx["relationship_gate"] = f"married_to_{record.get('character', '')}"
    else:
        ctx["heart_min"] = _num(sc.get("heart_min"))
        ctx["relationship_gate"] = _scalar(sc.get("relationship_gate"))

    # Optional fields
    variant_index = sc.get("variant_index")
    if variant_index is not None:
        ctx["variant_index"] = variant_index

    if "mood" in sc and sc["mood"] is not None:
        ctx["mood"] = sc["mood"]

    if "family_state" in sc and sc["family_state"] is not None:
        ctx["family_state"] = sc["family_state"]

    return ctx


# ---------------------------------------------------------------------------
# Retrieval / embedding text
# ---------------------------------------------------------------------------

def _build_embedding_text(
    record: dict[str, Any],
    embedding_text: str,
    scene_type: str = "daily",
) -> str:
    """Build the text used for embedding retrieval.

    Includes only semantic retrieval information – explicitly excludes
    control metadata (required_flags, route), source paths, hashes, etc.
    """
    sc = record.get("script_conditions", {})
    character = record.get("character", "any")
    dialogue_type = record.get("dialogue_type", "any")

    def _arr_str(val: Any) -> str:
        if val is None:
            return "any"
        if isinstance(val, list):
            return ", ".join(str(v) for v in val) if val else "any"
        return str(val)

    # Use marriage-aware relationship gate
    is_marriage = dialogue_type == "marriage_dialogue"
    if is_marriage:
        rel_gate = f"married_to_{character}"
    else:
        rel_gate = sc.get("relationship_gate", "any")

    parts: list[str] = [
        f"Character: {character}",
        f"Dialogue type: {dialogue_type}",
        f"Scene type: {scene_type}",
        f"Relationship: {rel_gate}",
        f"Season: {_arr_str(sc.get('season'))}",
        f"Weather: {_arr_str(sc.get('weather'))}",
    ]

    # Route — include if not "any" for better semantic retrieval
    route = record.get("control", {}).get("route", "any")
    if route and route != "any":
        parts.append(f"Route: {route}")

    # Scene
    if "scene" in sc and sc["scene"] is not None:
        parts.append(f"Scene: {sc['scene']}")

    # Time period
    if "time_period" in sc and sc["time_period"] is not None:
        parts.append(f"Time: {_arr_str(sc['time_period'])}")

    # Mood
    if "mood" in sc and sc["mood"] is not None:
        parts.append(f"Mood: {sc['mood']}")

    # Family state
    if "family_state" in sc and sc["family_state"] is not None:
        parts.append(f"Family state: {sc['family_state']}")

    parts.append(f"Dialogue: {embedding_text}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Control normalization
# ---------------------------------------------------------------------------

def _normalize_control(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure control has required_flags list and route string."""
    ctrl = record.get("control", {})
    if not isinstance(ctrl, dict):
        ctrl = {}
    return {
        "required_flags": ctrl.get("required_flags", []) or [],
        "route": ctrl.get("route", "any") or "any",
    }


# ---------------------------------------------------------------------------
# Core: canonical → processed records
# ---------------------------------------------------------------------------

def build_processed_record(
    canonical_record: dict[str, Any],
    lang: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Expand one canonical record for a single language.

    Parameters
    ----------
    canonical_record : dict
        A canonical dialogue record with ``texts``, ``script_conditions``, etc.
    lang : str
        The language code to extract (e.g. ``"en"``, ``"zh"``).
    embedding_model : str
        Model name for the ``indexing`` field.

    Returns
    -------
    dict
        A processed record.
    """
    canonical_id = canonical_record["canonical_id"]
    raw_text = canonical_record.get("texts", {}).get(lang, "")

    raw = _normalize_raw(raw_text)
    display = _normalize_display(raw_text)
    embedding = _normalize_embedding(raw_text)

    processed_id = f"{canonical_id}:{lang}"

    context = _build_context(canonical_record)
    control = _normalize_control(canonical_record)
    scene_type = _infer_scene_type(
        canonical_record.get("author_key", ""),
        canonical_record.get("dialogue_type", ""),
        canonical_record.get("script_conditions", {}),
    )

    # Hashes
    text_hash = _sha256(raw)
    retrieval_text = _build_embedding_text(canonical_record, embedding, scene_type=scene_type)
    retrieval_text_hash = _sha256(retrieval_text)

    indexing: dict[str, Any] = {
        "text_hash": text_hash,
        "retrieval_text_hash": retrieval_text_hash,
        "embedding_model": embedding_model,
        "vector_ref": f"{processed_id}:text",
        "_embedding_text": retrieval_text,
    }

    return {
        "processed_id": processed_id,
        "canonical_id": canonical_id,
        "author_key": canonical_record.get("author_key", ""),
        "character": canonical_record.get("character", ""),
        "dialogue_type": canonical_record.get("dialogue_type", ""),
        "lang": lang,
        "text": {
            "raw": raw,
            "display": display,
            "embedding": embedding,
        },
        "context": context,
        "control": control,
        "scene_type": scene_type,
        "quality": canonical_record.get("quality", {}),
        "indexing": indexing,
    }


def process_canonical_file(
    input_path: Path,
    output_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Read a canonical JSON file, expand to processed records, write output.

    Returns a summary dict with counts.
    """
    data = load_json(input_path)
    meta = data.get("dataset_meta", {})
    records = data.get("records", [])

    processed: list[dict[str, Any]] = []
    langs_seen: set[str] = set()

    for rec in records:
        texts = rec.get("texts", {})
        for lang in sorted(texts.keys()):
            langs_seen.add(lang)
            processed.append(
                build_processed_record(rec, lang, embedding_model=embedding_model)
            )

    output = {
        "dataset_meta": {
            **meta,
            "schema": "processed_dialogue_v1",
            "processed_from": str(input_path),
            "record_count": len(processed),
            "languages": sorted(langs_seen),
        },
        "records": processed,
    }

    save_json(output, output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "canonical_records": len(records),
        "processed_records": len(processed),
        "languages": sorted(langs_seen),
    }


# ---------------------------------------------------------------------------
# Load embedding model from config
# ---------------------------------------------------------------------------

def _load_embedding_model() -> str:
    """Try to read embedding model from index_config.yaml; fall back to default."""
    try:
        import yaml
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            return cfg.get("embedding", {}).get("model_name", DEFAULT_EMBEDDING_MODEL)
    except Exception:
        pass
    return DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Main: batch process all canonical files
# ---------------------------------------------------------------------------

def main() -> None:
    embedding_model = _load_embedding_model()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    canonical_files = sorted(CANONICAL_DIR.glob("*_canonical.json"))
    if not canonical_files:
        print(f"No canonical files found in {CANONICAL_DIR}")
        return

    total_processed = 0
    summaries: list[dict[str, Any]] = []

    for cf in canonical_files:
        character = cf.stem.replace("_canonical", "")
        output_path = PROCESSED_DIR / f"{character}_processed.json"
        summary = process_canonical_file(cf, output_path, embedding_model=embedding_model)
        summaries.append(summary)
        total_processed += summary["processed_records"]
        print(
            f"  {character}: {summary['canonical_records']} canonical → "
            f"{summary['processed_records']} processed "
            f"({', '.join(summary['languages'])})"
        )

    print(f"\nTotal processed records: {total_processed}")
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
