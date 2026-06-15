"""NPC configuration loader.

Reads ``npc/{name}.json`` and returns relationship-specific data.

Public API:
    load_npc_config   — returns a full dict (preferred)
    load_relationship_config — backward-compatible tuple (deprecated)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_examples(examples) -> str:
    """Format examples into prompt-ready text.

    Output format:
        - [场景标签] NPC 原台词或风格示例

    Tag values from native NPC JSON (like "rainy", "summer", "Mountain")
    are used directly as scene labels.  String-type examples (like Damon's
    single example per relationship) get no tag label.
    """
    if not examples:
        return ""

    # Normalize: single string → wrap in list
    if isinstance(examples, str):
        examples = [examples]

    lines: list[str] = []
    for item in examples:
        if isinstance(item, str):
            if item:
                lines.append(f"- {item}")
        elif isinstance(item, dict):
            tag = item.get("tag", "")
            content = item.get("content", "")
            if content:
                if tag and tag.lower() not in ("none", ""):
                    lines.append(f"- [{tag}] {content}")
                else:
                    lines.append(f"- {content}")

    return "\n".join(lines) if lines else ""


def _build_description(target_data: dict, relationship_status: str) -> str:
    description = target_data.get("description", "").strip()
    if description:
        return description

    fallback_status = (relationship_status or "stranger").replace("_", " ").strip()
    return f"Current relationship stage: {fallback_status}."


# ---------------------------------------------------------------------------
# Top-level fields to extract from NPC JSON
# ---------------------------------------------------------------------------

_PERSONA_FIELDS = (
    "character_type",
    "persona_core",
    "persona_background",
    "persona_growth",
    "speech_style",
    "mood_rules",
    "dialogue_constraints",
    "do",
    "dont",
)


# ---------------------------------------------------------------------------
# Public: load_npc_config (preferred)
# ---------------------------------------------------------------------------

def load_npc_config(npc_id: str, relationship_status: str) -> dict:
    """Load full NPC config including persona fields and relationship-specific data.

    Returns a dict with:
    - Top-level persona fields: character_type, persona_core, persona_background,
      persona_growth, speech_style, mood_rules, dialogue_constraints, do, dont
    - Relationship-specific fields: description, instruction, gift, static_examples

    Missing fields use sensible defaults so callers never need to check for
    key existence.
    """
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / f"{npc_id}.json"

    # Defaults
    config: dict = {
        "character_type": "native",
        "persona_core": "",
        "persona_background": None,
        "persona_growth": None,
        "speech_style": None,
        "mood_rules": None,
        "dialogue_constraints": None,
        "do": [],
        "dont": [],
        "description": "",
        "instruction": "",
        "gift": "",
        "static_examples": "",
    }

    if not file_path.exists():
        print(f"⚠️ [Config] 未找到 {npc_id} 的配置文件，使用默认值。")
        return config

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ [Config] 读取 {npc_id} 配置出错: {e}")
        return config

    # Top-level persona fields
    for key in _PERSONA_FIELDS:
        if key in data:
            config[key] = data[key]

    # Relationship-specific fields
    data_map = data.get("relationship_map", {})
    key = (relationship_status or "stranger").lower()

    if key not in data_map:
        target_data = data_map.get("stranger", {})
    else:
        target_data = data_map[key]

    config["description"] = _build_description(target_data, key)
    config["instruction"] = target_data.get("instruction", "")
    config["gift"] = target_data.get("gift", "")
    config["static_examples"] = _format_examples(
        target_data.get("examples", target_data.get("example", []))
    )

    return config


# ---------------------------------------------------------------------------
# Public: load_relationship_config (backward-compatible, deprecated)
# ---------------------------------------------------------------------------

def load_relationship_config(npc_id: str, relationship_status: str) -> Tuple[str, str, str, str]:
    """Read NPC config and return (description, instruction, gift, examples).

    .. deprecated::
        Use :func:`load_npc_config` instead.  This function is kept for
        backward compatibility only.
    """
    config = load_npc_config(npc_id, relationship_status)
    return (
        config["description"],
        config["instruction"],
        config["gift"],
        config["static_examples"],
    )
