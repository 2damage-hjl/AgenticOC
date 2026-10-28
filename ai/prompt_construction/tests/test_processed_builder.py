"""Tests for the processed dialogue builder.

Covers:
- Multilingual expansion (canonical → per-language processed records)
- processed_id format
- Text normalization (raw / display / embedding)
- Hash generation (text_hash, retrieval_text_hash)
- vector_ref format
- Control normalization (required_flags, route defaults)
- Exclusion of control metadata from embedding / retrieval text
- Marriage and Resort flag preservation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prompt_construction.scripts.build_processed_dialogues import (
    build_processed_record,
    _normalize_display,
    _normalize_embedding,
    _normalize_raw,
    _build_context,
    _build_embedding_text,
    _normalize_control,
    _sha256,
    _infer_scene_type,
    DEFAULT_EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_canonical_record(**overrides) -> dict:
    """Create a canonical record with sensible defaults."""
    base = {
        "canonical_id": "Dialogue/general_dialogue/Abigail:winter_Wed2",
        "author_key": "winter_Wed2",
        "character": "Abigail",
        "dialogue_type": "general_dialogue",
        "texts": {
            "en": "It's so cold out...$s#$b#But I kind of like it.",
            "zh": "外面好冷啊……$s#$b#不过我挺喜欢的。",
        },
        "source": {
            "source_type": "verified_game_asset",
            "collection": "Dialogue",
            "files": {
                "en": "data/game_scripts/Dialogue/Abigail.json",
                "zh": "data/game_scripts/Dialogue/Abigail.zh-CN.json",
            },
        },
        "script_conditions": {
            "season": ["winter"],
            "day_of_week": ["Wed"],
            "day_of_month": ["any"],
            "heart_min": 2,
            "relationship_gate": "heart_min_2",
            "weather": ["any"],
            "special_key_type": "generic_dialogue",
            "variant_index": None,
            "parse_warnings": [],
        },
        "control": {
            "required_flags": [],
            "route": "any",
        },
        "quality": {
            "source_confidence": "high",
            "script_parse_confidence": "high",
        },
    }
    base.update(overrides)
    return base


def _make_marriage_record(**overrides) -> dict:
    """Create a marriage canonical record."""
    base = {
        "canonical_id": "Dialogue/marriage_dialogue/Abigail:Indoor_Day_0",
        "author_key": "Indoor_Day_0",
        "character": "Abigail",
        "dialogue_type": "marriage_dialogue",
        "texts": {
            "en": "I always loved this place...$h",
            "zh": "我一直很喜欢这个地方……$h",
        },
        "source": {
            "source_type": "verified_game_asset",
            "collection": "Dialogue",
            "files": {
                "en": "data/game_scripts/Dialogue/MarriageDialogueAbigail.json",
                "zh": "data/game_scripts/Dialogue/MarriageDialogueAbigail.zh-CN.json",
            },
        },
        "script_conditions": {
            "season": ["any"],
            "day_of_week": ["any"],
            "day_of_month": ["any"],
            "heart_min": 10,
            "relationship_gate": "married",
            "weather": ["any"],
            "special_key_type": "marriage_dialogue",
            "variant_index": 0,
            "parse_warnings": [],
        },
        "control": {
            "required_flags": ["relationship.married_to.Abigail"],
            "route": "any",
        },
        "quality": {
            "source_confidence": "high",
            "script_parse_confidence": "high",
        },
    }
    base.update(overrides)
    return base


# ===================================================================
# Test: Multilingual expansion
# ===================================================================

class TestMultilingualExpansion:
    """One canonical record with en + zh → two processed records."""

    def test_two_records_from_two_langs(self):
        rec = _make_canonical_record()
        en = build_processed_record(rec, "en")
        zh = build_processed_record(rec, "zh")
        assert en["lang"] == "en"
        assert zh["lang"] == "zh"

    def test_processed_id_format(self):
        rec = _make_canonical_record()
        en = build_processed_record(rec, "en")
        assert en["processed_id"] == "Dialogue/general_dialogue/Abigail:winter_Wed2:en"

    def test_processed_id_contains_lang(self):
        rec = _make_canonical_record()
        zh = build_processed_record(rec, "zh")
        assert zh["processed_id"].endswith(":zh")

    def test_canonical_id_preserved(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["canonical_id"] == rec["canonical_id"]


# ===================================================================
# Test: Text normalization
# ===================================================================

class TestTextNormalization:
    """Verify raw / display / embedding text normalization."""

    def test_raw_preserves_stardew_tokens(self):
        text = "Hello @!$h#$b#Bye.$s"
        result = _normalize_raw(text)
        assert "$h" in result
        assert "#$b#" in result
        assert "@" in result

    def test_display_replaces_player_token(self):
        text = "Hello @!$h#$b#Bye.$s"
        result = _normalize_display(text)
        assert "{player_name}" in result
        assert "@" not in result

    def test_display_strips_dollar_tokens(self):
        text = "Hello$h#$b#Bye.$s"
        result = _normalize_display(text)
        assert "$h" not in result
        assert "$s" not in result
        assert "#$b#" not in result

    def test_display_converts_break_token(self):
        text = "Line1#$b#Line2"
        result = _normalize_display(text)
        assert "#$b#" not in result
        assert "\n" in result

    def test_embedding_replaces_at_with_player(self):
        text = "Hello @!$h"
        result = _normalize_embedding(text)
        assert "player" in result
        assert "@" not in result

    def test_embedding_strips_dollar_tokens(self):
        text = "Hello$h#$b#Bye$s"
        result = _normalize_embedding(text)
        assert "$h" not in result
        assert "$s" not in result
        assert "#$b#" not in result

    def test_embedding_break_becomes_space(self):
        text = "Line1#$b#Line2"
        result = _normalize_embedding(text)
        assert "#$b#" not in result
        assert "Line1" in result
        assert "Line2" in result


# ===================================================================
# Test: Token cleanup ($query, %spouse, #$e#, | separator)
# ===================================================================

class TestTokenCleanup:
    """Verify raw game tokens are cleaned from display/embedding text."""

    def test_display_strips_query_token(self):
        text = "$query PLAYER_NPC_RELATIONSHIP current any married roommate#Hey... How are things?"
        result = _normalize_display(text)
        assert "PLAYER_NPC_RELATIONSHIP" not in result
        assert "$query" not in result
        assert "current any married roommate" not in result
        assert "Hey" in result

    def test_embedding_strips_query_token(self):
        text = "$query PLAYER_NPC_RELATIONSHIP current any married roommate#Hey... How are things?"
        result = _normalize_embedding(text)
        assert "PLAYER_NPC_RELATIONSHIP" not in result
        assert "$query" not in result
        assert "Hey" in result

    def test_display_replaces_spouse_token(self):
        text = "Congratulations, @. This is a big step for you and %spouse."
        result = _normalize_display(text)
        assert "%spouse" not in result
        assert "{spouse_name}" in result

    def test_embedding_replaces_spouse_token(self):
        text = "Congratulations, @. This is a big step for you and %spouse."
        result = _normalize_embedding(text)
        assert "%spouse" not in result
        assert "spouse" in result

    def test_display_strips_e_marker(self):
        text = "Same old routine, huh?#$e#Yeah, I know a thing or two about that..."
        result = _normalize_display(text)
        assert "#$e#" not in result
        assert "#$e" not in result

    def test_embedding_strips_e_marker(self):
        text = "Same old routine, huh?#$e#Yeah, I know a thing or two about that..."
        result = _normalize_embedding(text)
        assert "#$e#" not in result
        assert "#$e" not in result

    def test_display_keeps_first_part_of_pipe(self):
        text = "First response|Second response"
        result = _normalize_display(text)
        assert "First response" in result
        assert "Second response" not in result

    def test_embedding_keeps_first_part_of_pipe(self):
        text = "First response|Second response"
        result = _normalize_embedding(text)
        assert "First response" in result
        assert "Second response" not in result

    def test_full_query_cleanup_zh(self):
        """Real-world example: Chinese text with $query token."""
        text = "$query PLAYER_NPC_RELATIONSHIP current any married roommate#嘿……最近农场怎么样？#$e#每天都是一成不变的日常是不是？我也略有这种感受……|你能想象我这样的人住在农场里吗？#$e#虽然很荒唐，但最近我一直在考虑着。"
        result = _normalize_display(text)
        assert "PLAYER_NPC_RELATIONSHIP" not in result
        assert "$query" not in result
        assert "#$e" not in result
        assert "嘿" in result

    def test_full_query_cleanup_embedding_zh(self):
        """Real-world example: Chinese embedding with $query token."""
        text = "$query PLAYER_NPC_RELATIONSHIP current any married roommate#嘿……最近农场怎么样？#$e#每天都是一成不变的日常是不是？我也略有这种感受……|你能想象我这样的人住在农场里吗？#$e#虽然很荒唐，但最近我一直在考虑着。"
        result = _normalize_embedding(text)
        assert "PLAYER_NPC_RELATIONSHIP" not in result
        assert "$query" not in result

    def test_crlf_normalized(self):
        text = "Line1\r\nLine2"
        result = _normalize_raw(text)
        assert "\r\n" not in result
        assert "\n" in result

    def test_whitespace_compressed(self):
        text = "Hello    world"
        result = _normalize_raw(text)
        assert result == "Hello world"


# ===================================================================
# Test: Hash generation
# ===================================================================

class TestHashGeneration:

    def test_text_hash_generated(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["indexing"]["text_hash"]
        assert len(p["indexing"]["text_hash"]) == 64  # SHA256 hex

    def test_text_hash_is_sha256(self):
        raw = "Hello world"
        h = _sha256(raw)
        assert len(h) == 64

    def test_text_hash_deterministic(self):
        rec = _make_canonical_record()
        p1 = build_processed_record(rec, "en")
        p2 = build_processed_record(rec, "en")
        assert p1["indexing"]["text_hash"] == p2["indexing"]["text_hash"]

    def test_retrieval_text_hash_generated(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["indexing"]["retrieval_text_hash"]
        assert len(p["indexing"]["retrieval_text_hash"]) == 64

    def test_different_langs_different_text_hash(self):
        rec = _make_canonical_record()
        en = build_processed_record(rec, "en")
        zh = build_processed_record(rec, "zh")
        assert en["indexing"]["text_hash"] != zh["indexing"]["text_hash"]

    def test_vector_ref_generated(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["indexing"]["vector_ref"] == "Dialogue/general_dialogue/Abigail:winter_Wed2:en:text"


# ===================================================================
# Test: Control normalization
# ===================================================================

class TestControlNormalization:

    def test_required_flags_preserved(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert "relationship.married_to.Abigail" in p["control"]["required_flags"]

    def test_route_preserved(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["control"]["route"] == "any"

    def test_missing_required_flags_defaults_to_empty_list(self):
        rec = _make_canonical_record()
        del rec["control"]
        p = build_processed_record(rec, "en")
        assert p["control"]["required_flags"] == []

    def test_missing_route_defaults_to_any(self):
        rec = _make_canonical_record()
        del rec["control"]
        p = build_processed_record(rec, "en")
        assert p["control"]["route"] == "any"

    def test_none_required_flags_defaults_to_empty_list(self):
        rec = _make_canonical_record(control={"required_flags": None, "route": "any"})
        p = build_processed_record(rec, "en")
        assert p["control"]["required_flags"] == []

    def test_none_route_defaults_to_any(self):
        rec = _make_canonical_record(control={"required_flags": [], "route": None})
        p = build_processed_record(rec, "en")
        assert p["control"]["route"] == "any"

    def test_resort_required_flag_preserved(self):
        rec = _make_canonical_record(
            control={"required_flags": ["world.ginger_island.beach_resort.opened"], "route": "any"}
        )
        p = build_processed_record(rec, "en")
        assert "world.ginger_island.beach_resort.opened" in p["control"]["required_flags"]

    def test_marriage_required_flag_preserved(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert "relationship.married_to.Abigail" in p["control"]["required_flags"]


# ===================================================================
# Test: Embedding text excludes control metadata
# ===================================================================

class TestEmbeddingTextExcludesControl:

    def test_required_flags_not_in_embedding_text(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        emb_text = p["indexing"]["_embedding_text"]
        assert "required_flags" not in emb_text
        assert "relationship.married_to.Abigail" not in emb_text

    def test_route_not_in_embedding_text(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        emb_text = p["indexing"]["_embedding_text"]
        assert "route" not in emb_text.lower() or "route" not in emb_text
        # More precisely: no "route:" line
        lines = emb_text.split("\n")
        for line in lines:
            assert not line.startswith("Route:")

    def test_required_flags_not_in_text_embedding_field(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert "required_flags" not in p["text"]["embedding"]

    def test_route_not_in_text_embedding_field(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "route" not in p["text"]["embedding"]

    def test_source_path_not_in_embedding_text(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        emb_text = p["indexing"]["_embedding_text"]
        assert "game_scripts" not in emb_text

    def test_text_hash_not_in_embedding_text(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        emb_text = p["indexing"]["_embedding_text"]
        assert "text_hash" not in emb_text


# ===================================================================
# Test: Context construction
# ===================================================================

class TestContextConstruction:

    def test_season_from_script_conditions(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["season"] == ["winter"]

    def test_heart_min_numeric(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["heart_min"] == 2

    def test_heart_min_null(self):
        rec = _make_canonical_record()
        rec["script_conditions"]["heart_min"] = None
        p = build_processed_record(rec, "en")
        assert p["context"]["heart_min"] is None

    def test_relationship_gate(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["relationship_gate"] == "heart_min_2"

    def test_marriage_relationship_gate(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["relationship_gate"] == "married_to_Abigail"

    def test_variant_index_present_for_marriage(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["variant_index"] == 0

    def test_variant_index_absent_for_general(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "variant_index" not in p["context"]

    def test_default_arrays_use_any(self):
        rec = _make_canonical_record()
        rec["script_conditions"]["weather"] = None
        p = build_processed_record(rec, "en")
        assert p["context"]["weather"] == ["any"]

    def test_source_key_equals_author_key(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["source_key"] == "winter_Wed2"

    def test_source_key_not_file_path(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "/" not in p["context"]["source_key"]
        assert "\\" not in p["context"]["source_key"]
        assert ".json" not in p["context"]["source_key"]

    def test_marriage_heart_min_is_none(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["heart_min"] is None

    def test_marriage_relationship_gate_format(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["relationship_gate"] == "married_to_Abigail"

    def test_marriage_variant_index_preserved(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert p["context"]["variant_index"] == 0

    def test_marriage_required_flags_in_control(self):
        rec = _make_marriage_record()
        p = build_processed_record(rec, "en")
        assert "relationship.married_to.Abigail" in p["control"]["required_flags"]


# ===================================================================
# Test: Embedding model and indexing
# ===================================================================

class TestIndexing:

    def test_embedding_model_default(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["indexing"]["embedding_model"] == DEFAULT_EMBEDDING_MODEL

    def test_vector_ref_format(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["indexing"]["vector_ref"] == f"{p['processed_id']}:text"

    def test_embedding_text_in_indexing(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "_embedding_text" in p["indexing"]
        assert "Character: Abigail" in p["indexing"]["_embedding_text"]

    def test_embedding_text_contains_dialogue(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "Dialogue:" in p["indexing"]["_embedding_text"]


# ===================================================================
# Test: Full processed record structure
# ===================================================================

class TestProcessedRecordStructure:

    def test_all_required_fields_present(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        required_fields = [
            "processed_id", "canonical_id", "author_key", "character",
            "dialogue_type", "lang", "text", "context", "control",
            "scene_type", "quality", "indexing",
        ]
        for field in required_fields:
            assert field in p, f"Missing field: {field}"

    def test_text_subfields(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "raw" in p["text"]
        assert "display" in p["text"]
        assert "embedding" in p["text"]

    def test_indexing_subfields(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "text_hash" in p["indexing"]
        assert "retrieval_text_hash" in p["indexing"]
        assert "embedding_model" in p["indexing"]
        assert "vector_ref" in p["indexing"]

    def test_quality_copied_from_canonical(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["quality"] == rec["quality"]


# ===================================================================
# Test: Scene type inference
# ===================================================================

class TestSceneTypeInference:
    """Verify _infer_scene_type correctly classifies dialogue records."""

    def test_marriage_dialogue_type(self):
        result = _infer_scene_type("Indoor_Day_0", "marriage_dialogue", {})
        assert result == "marriage"

    def test_birthday_gift_key(self):
        result = _infer_scene_type("AcceptBirthdayGift_Positive", "general_dialogue", {})
        assert result == "birthday"

    def test_birthday_reject_key(self):
        result = _infer_scene_type("RejectBirthdayGift", "general_dialogue", {})
        assert result == "birthday"

    def test_rain_key(self):
        result = _infer_scene_type("rain", "general_dialogue", {"special_key_type": "weather_rain"})
        assert result == "weather"

    def test_resort_key(self):
        result = _infer_scene_type("Resort", "general_dialogue", {"special_key_type": "ginger_island_resort"})
        assert result == "location"

    def test_cc_key(self):
        result = _infer_scene_type("cc_Bridge", "general_dialogue", {"special_key_type": "community_center_event"})
        assert result == "special"

    def test_gift_key(self):
        result = _infer_scene_type("AcceptGift", "general_dialogue", {})
        assert result == "gift"

    def test_seasonal_key(self):
        result = _infer_scene_type(
            "winter_Wed2", "general_dialogue",
            {"season": ["winter"], "special_key_type": "generic_dialogue"},
        )
        assert result == "seasonal"

    def test_daily_key(self):
        result = _infer_scene_type(
            "4", "general_dialogue",
            {"season": ["any"], "special_key_type": "generic_dialogue"},
        )
        assert result == "daily"

    def test_festival_date_specific(self):
        result = _infer_scene_type(
            "spring_13", "general_dialogue",
            {"season": ["spring"], "day_of_month": ["13"], "special_key_type": "date_specific_dialogue"},
        )
        assert result == "festival"

    def test_non_festival_date_specific(self):
        result = _infer_scene_type(
            "spring_5", "general_dialogue",
            {"season": ["spring"], "day_of_month": ["5"], "special_key_type": "date_specific_dialogue"},
        )
        assert result == "seasonal"

    def test_breakup_key(self):
        result = _infer_scene_type("breakUp", "general_dialogue", {})
        assert result == "special"

    def test_divorced_key(self):
        result = _infer_scene_type("divorced", "general_dialogue", {})
        assert result == "special"

    def test_scene_type_in_processed_record(self):
        rec = _make_canonical_record(author_key="AcceptBirthdayGift_Positive")
        p = build_processed_record(rec, "en")
        assert p["scene_type"] == "birthday"

    def test_scene_type_daily_in_processed_record(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert p["scene_type"] == "seasonal"

    def test_scene_type_in_embedding_text(self):
        rec = _make_canonical_record()
        p = build_processed_record(rec, "en")
        assert "Scene type:" in p["indexing"]["_embedding_text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
