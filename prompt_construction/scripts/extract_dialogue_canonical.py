"""Canonical dialogue extraction for Stardew Valley.

Reads raw game scripts from ``data/game_scripts/Dialogue/``, detects
language and character from filenames, parses author keys into deterministic
script conditions, aligns multilingual texts by canonical_id, and outputs
structured canonical records.

Outputs:
    data/canonical/dialogue/general.json     – general_dialogue records
    data/canonical/dialogue/marriage.json     – marriage_dialogue records
    data/canonical/dialogue/all_dialogues.json – merged and sorted

Does NOT modify anything under data/game_scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Local imports – add parent to path so utils is importable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.json_io import load_json, save_json
from utils.text_normalize import normalize_dialogue_text
from utils.dialogue_key_parser import (
    parse_dialogue_key,
    extract_character_from_shared_marriage_key,
    MARRIAGE_CHARACTERS,
)

# ---------------------------------------------------------------------------
# $d conditional text splitting
# ---------------------------------------------------------------------------
# Stardew Valley uses ``$d <condition>#<text_A>|<text_B>`` to embed
# route-conditional text.  We split these into separate canonical records
# so each branch gets its own route tag.
#
# Supported conditions:
#   $d joja#  → Joja route active vs not
#   $d cc#    → Community Center completed vs not
# ---------------------------------------------------------------------------

_ROUTE_COND_RE = re.compile(
    r"^\$d\s+(joja|cc)\s*#(.*?)\|(.*)",
    re.IGNORECASE | re.DOTALL,
)

ROUTE_BRANCH_MAP = {
    # condition → (route for text_A, route for text_B)
    "joja": ("joja", "non_joja"),
    "cc":   ("community_center", "non_cc"),
}


def split_route_conditional(
    text: str, author_key: str, character: str
) -> list[tuple[str, str | None]]:
    """If *text* starts with a ``$d`` route conditional, split it.

    Returns a list of ``(text, route_or_None)`` tuples.  If no conditional
    is found, returns ``[(text, None)]``.
    """
    m = _ROUTE_COND_RE.match(text)
    if not m:
        return [(text, None)]

    condition = m.group(1).lower()
    text_a = m.group(2).strip()
    text_b = m.group(3).strip()

    if condition not in ROUTE_BRANCH_MAP:
        return [(text, None)]

    route_a, route_b = ROUTE_BRANCH_MAP[condition]
    return [(text_a, route_a), (text_b, route_b)]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = PROJECT_ROOT / "data" / "game_scripts" / "Dialogue"
OUTPUT_DIR = PROJECT_ROOT / "data" / "canonical" / "dialogue"

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

LANG_ALIASES: dict[str, str] = {
    "en": "en",
    "english": "en",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "chinese": "zh",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kr": "ko",
    "korean": "ko",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "pt": "pt",
    "ru": "ru",
    "it": "it",
    "tr": "tr",
    "hu": "hu",
}


def normalize_lang(raw: str | None) -> str:
    if not raw:
        return "en"
    key = raw.strip().lower().replace("_", "-")
    if key in LANG_ALIASES:
        return LANG_ALIASES[key]
    primary = key.split("-")[0]
    if primary in LANG_ALIASES:
        return LANG_ALIASES[primary]
    return primary


def infer_lang_from_path(path: Path) -> str:
    """Detect language from filename suffix (e.g. Abigail.zh-CN.json → zh)."""
    stem_parts = path.stem.split(".")
    if len(stem_parts) >= 2:
        possible_lang = normalize_lang(stem_parts[-1])
        valid_langs = set(LANG_ALIASES.values())
        if possible_lang in valid_langs:
            return possible_lang
    return "en"


# ---------------------------------------------------------------------------
# Character detection
# ---------------------------------------------------------------------------

def is_marriage_file(stem: str) -> bool:
    """Return True if the filename stem indicates a marriage dialogue file."""
    return stem.startswith("MarriageDialogue")


def infer_character_from_path(path: Path) -> str:
    """Extract character name from a general dialogue filename."""
    stem_parts = path.stem.split(".")
    base = stem_parts[0]  # e.g. "Abigail" or "MarriageDialogueAbigail"
    if len(stem_parts) >= 2:
        possible_lang = normalize_lang(stem_parts[-1])
        valid_langs = set(LANG_ALIASES.values())
        if possible_lang in valid_langs:
            base = ".".join(stem_parts[:-1])
    return base


def extract_character_from_marriage_filename(stem: str) -> str | None:
    """For MarriageDialogue{Name} files, return {Name}. Returns None for the shared file."""
    # MarriageDialogueAbigail -> Abigail
    # MarriageDialogue -> None (shared)
    if stem == "MarriageDialogue":
        return None
    for char in MARRIAGE_CHARACTERS:
        if stem == f"MarriageDialogue{char}":
            return char
    return None


# ---------------------------------------------------------------------------
# Dialogue map validation
# ---------------------------------------------------------------------------

def is_dialogue_map(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    values = list(data.values())
    string_like = sum(isinstance(v, str) for v in values)
    return string_like / max(len(values), 1) >= 0.7


# ---------------------------------------------------------------------------
# Canonical ID
# ---------------------------------------------------------------------------

def make_canonical_id(dialogue_type: str, character: str, author_key: str) -> str:
    return f"Dialogue/{dialogue_type}/{character}:{author_key}"


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _safe_relative_to(path: Path, root: Path) -> str:
    """Return path relative to root as a string; fall back to str(path)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_record(
    canonical_id: str,
    author_key: str,
    character: str,
    dialogue_type: str,
    parsed: dict[str, Any],
    text_lang: str,
    text: str,
    file_path: Path,
    route: str | None = None,
) -> dict[str, Any]:
    """Create or partially populate a canonical record."""
    script_conditions = parsed["script_conditions"]
    key_required_flags = parsed.get("key_required_flags", [])

    # Build required_flags
    required_flags: list[str] = list(key_required_flags)
    if dialogue_type == "marriage_dialogue":
        flag = f"relationship.married_to.{character}"
        if flag not in required_flags:
            required_flags.append(flag)

    script_conditions["special_key_type"] = (
        "marriage_dialogue" if dialogue_type == "marriage_dialogue"
        and script_conditions["special_key_type"] == "generic_dialogue"
        else script_conditions["special_key_type"]
    )

    # Route: explicit param > key_route from parser > "any"
    if route is None:
        route = parsed.get("key_route", "any")

    return {
        "canonical_id": canonical_id,
        "author_key": author_key,
        "character": character,
        "dialogue_type": dialogue_type,
        "texts": {text_lang: text},
        "source": {
            "source_type": "verified_game_asset",
            "collection": "Dialogue",
            "files": {text_lang: _safe_relative_to(file_path, PROJECT_ROOT)},
        },
        "script_conditions": script_conditions,
        "control": {
            "required_flags": required_flags,
            "route": route,
        },
        "quality": {
            "source_confidence": "high",
            "script_parse_confidence": (
                "medium" if script_conditions.get("parse_warnings") else "high"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extract canonical records from all Dialogue files.

    Returns
    -------
    (general_records, marriage_records, report)
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Dialogue input directory not found: {INPUT_DIR}")

    files = sorted(INPUT_DIR.rglob("*.json"))

    # {canonical_id: record} for deduplication / multilingual merge
    general: dict[str, dict[str, Any]] = {}
    marriage: dict[str, dict[str, Any]] = {}
    languages_seen: set[str] = set()

    report = {
        "total_files": len(files),
        "general_files": 0,
        "marriage_files": 0,
        "general_records": 0,
        "marriage_records": 0,
        "characters": {},
        "warnings": [],
        "duplicate_overwrites": [],
        "missing_languages": {},
        "languages_found": [],
    }

    for file_path in files:
        stem_parts = file_path.stem.split(".")
        base_stem = stem_parts[0]

        lang = infer_lang_from_path(file_path)
        languages_seen.add(lang)
        is_marry = is_marriage_file(base_stem)

        try:
            data = load_json(file_path)
        except Exception as e:
            report["warnings"].append(f"Failed to load {file_path.name}: {e}")
            continue

        if not is_dialogue_map(data):
            report["warnings"].append(f"Skipped non-dialogue map: {file_path.name}")
            continue

        if is_marry:
            report["marriage_files"] += 1
            _process_marriage_file(file_path, base_stem, lang, data, marriage, report)
        else:
            report["general_files"] += 1
            _process_general_file(file_path, base_stem, lang, data, general, report)

    general_list = sorted(general.values(), key=lambda r: r["canonical_id"])
    marriage_list = sorted(marriage.values(), key=lambda r: r["canonical_id"])

    report["general_records"] = len(general_list)
    report["marriage_records"] = len(marriage_list)

    # Compute missing languages across all records
    report["languages_found"] = sorted(languages_seen)
    all_records_dict = {**general, **marriage}

    if languages_seen:
        for cid, rec in all_records_dict.items():
            present = set(rec["texts"].keys())
            missing = sorted(languages_seen - present)
            if missing:
                report["missing_languages"][cid] = missing

    return general_list, marriage_list, report


def _process_general_file(
    file_path: Path,
    base_stem: str,
    lang: str,
    data: dict[str, Any],
    records: dict[str, dict[str, Any]],
    report: dict[str, Any],
) -> None:
    character = base_stem
    dialogue_type = "general_dialogue"

    for author_key, raw_text in data.items():
        text = normalize_dialogue_text(raw_text)
        if not text:
            continue

        parsed = parse_dialogue_key(author_key, is_marriage=False)

        # Split $d route-conditional text into separate branches
        branches = split_route_conditional(text, author_key, character)

        for branch_text, branch_route in branches:
            # Build canonical_id — append route suffix for conditional branches
            if branch_route is not None:
                effective_key = f"{author_key}_{branch_route}"
            else:
                effective_key = author_key

            canonical_id = make_canonical_id(dialogue_type, character, effective_key)

            if canonical_id in records:
                # Duplicate canonical_id + lang → record warning, overwrite
                if lang in records[canonical_id]["texts"]:
                    report["duplicate_overwrites"].append({
                        "canonical_id": canonical_id,
                        "lang": lang,
                        "file": str(file_path),
                        "author_key": author_key,
                    })
                records[canonical_id]["texts"][lang] = branch_text
                records[canonical_id]["source"]["files"][lang] = _safe_relative_to(
                    file_path, PROJECT_ROOT
                )
            else:
                records[canonical_id] = build_record(
                    canonical_id=canonical_id,
                    author_key=effective_key,
                    character=character,
                    dialogue_type=dialogue_type,
                    parsed=parsed,
                    text_lang=lang,
                    text=branch_text,
                    file_path=file_path,
                    route=branch_route,
                )

    if character not in report["characters"]:
        report["characters"][character] = {"general": 0, "marriage": 0}
    report["characters"][character]["general"] = len(
        [r for r in records.values() if r["character"] == character]
    )


def _process_marriage_file(
    file_path: Path,
    base_stem: str,
    lang: str,
    data: dict[str, Any],
    records: dict[str, dict[str, Any]],
    report: dict[str, Any],
) -> None:
    character_from_file = extract_character_from_marriage_filename(base_stem)
    is_shared = character_from_file is None

    for author_key, raw_text in data.items():
        text = normalize_dialogue_text(raw_text)
        if not text:
            continue

        if is_shared:
            # Shared MarriageDialogue.json: character may be in the key suffix
            char, clean_key = extract_character_from_shared_marriage_key(author_key)
            if char is None:
                # Generic key (e.g. NoBed_0) – assign to "Shared"
                char = "Shared"
            character = char
            effective_author_key = author_key  # preserve original key
        else:
            character = character_from_file
            effective_author_key = author_key

        dialogue_type = "marriage_dialogue"
        parsed = parse_dialogue_key(effective_author_key, is_marriage=True)

        # Split $d route-conditional text into separate branches
        branches = split_route_conditional(text, effective_author_key, character)

        for branch_text, branch_route in branches:
            if branch_route is not None:
                branch_key = f"{effective_author_key}_{branch_route}"
            else:
                branch_key = effective_author_key

            canonical_id = make_canonical_id(dialogue_type, character, branch_key)

            if canonical_id in records:
                if lang in records[canonical_id]["texts"]:
                    report["duplicate_overwrites"].append({
                        "canonical_id": canonical_id,
                        "lang": lang,
                        "file": str(file_path),
                        "author_key": effective_author_key,
                    })
                records[canonical_id]["texts"][lang] = branch_text
                records[canonical_id]["source"]["files"][lang] = _safe_relative_to(
                    file_path, PROJECT_ROOT
                )
            else:
                records[canonical_id] = build_record(
                    canonical_id=canonical_id,
                    author_key=branch_key,
                    character=character,
                    dialogue_type=dialogue_type,
                    parsed=parsed,
                    text_lang=lang,
                    text=branch_text,
                    file_path=file_path,
                    route=branch_route,
                )

    if character_from_file and character_from_file != "Shared":
        if character_from_file not in report["characters"]:
            report["characters"][character_from_file] = {"general": 0, "marriage": 0}
        report["characters"][character_from_file]["marriage"] += len(data)


# ---------------------------------------------------------------------------
# Per-character file output
# ---------------------------------------------------------------------------

def _write_per_character_files(
    all_records: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write per-character canonical JSON files (v2 schema) to output_dir.

    Each character gets a single file ``{Character}_canonical.json`` that
    contains **both** general_dialogue and marriage_dialogue records for
    that character.  For example, Abigail's general dialogues and marriage
    dialogues are merged into ``Abigail_canonical.json``.
    """
    from collections import defaultdict

    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_records:
        by_character[rec["character"]].append(rec)

    for character, records in sorted(by_character.items()):
        records_sorted = sorted(records, key=lambda r: r["canonical_id"])

        # Count dialogue types for metadata
        type_counts: dict[str, int] = {}
        for rec in records_sorted:
            dt = rec["dialogue_type"]
            type_counts[dt] = type_counts.get(dt, 0) + 1

        output = {
            "dataset_meta": {
                "schema": "canonical_dialogue_v2",
                "source_root": "data/game_scripts/Dialogue",
                "collection": "Dialogue",
                "character": character,
                "record_count": len(records_sorted),
                "dialogue_type_counts": type_counts,
            },
            "records": records_sorted,
        }
        filename = f"{character}_canonical.json"
        save_json(output, output_dir / filename)


def _cleanup_old_character_files(output_dir: Path) -> None:
    """Remove old per-character files with outdated naming conventions.

    Deletes:
      - ``{Character}_dialogue_canonical.json`` (old name, separated by type)
      - ``MarriageDialogue{Character}_dialogue_canonical.json`` (old marriage-only)
      - ``MarriageDialogue_dialogue_canonical.json`` (old shared marriage)
      - ``dialogue_canonical.json`` (old all-in-one file)
    """
    old_patterns = [
        "*_dialogue_canonical.json",  # e.g. Abigail_dialogue_canonical.json
    ]
    removed: list[str] = []
    for pattern in old_patterns:
        for old_file in output_dir.glob(pattern):
            old_file.unlink()
            removed.append(old_file.name)
    if removed:
        print(f"  Cleaned up {len(removed)} old per-character file(s): {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    general_records, marriage_records, report = extract_records()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    general_output = {
        "dataset_meta": {
            "schema": "canonical_dialogue_v2",
            "dialogue_type": "general_dialogue",
            "record_count": len(general_records),
        },
        "records": general_records,
    }
    marriage_output = {
        "dataset_meta": {
            "schema": "canonical_dialogue_v2",
            "dialogue_type": "marriage_dialogue",
            "record_count": len(marriage_records),
        },
        "records": marriage_records,
    }
    all_records = sorted(
        general_records + marriage_records,
        key=lambda r: r["canonical_id"],
    )
    all_output = {
        "dataset_meta": {
            "schema": "canonical_dialogue_v2",
            "dialogue_type": "mixed",
            "record_count": len(all_records),
        },
        "records": all_records,
    }

    save_json(general_output, OUTPUT_DIR / "general.json")
    save_json(marriage_output, OUTPUT_DIR / "marriage.json")
    save_json(all_output, OUTPUT_DIR / "all_dialogues.json")

    # Output per-character files to data/canonical/
    # Each {Character}_canonical.json merges general + marriage records
    canonical_dir = PROJECT_ROOT / "data" / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    _write_per_character_files(general_records + marriage_records, canonical_dir)

    # Clean up old per-character files with outdated naming
    _cleanup_old_character_files(canonical_dir)

    # Save extraction report
    report_dir = OUTPUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, report_dir / "extraction_report.json")

    print(f"General dialogue: {len(general_records)} records")
    print(f"Marriage dialogue: {len(marriage_records)} records")
    print(f"Total: {len(all_records)} records")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Per-character files: {canonical_dir}")
    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"][:10]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
