"""Tests for example_selector — text dedup, scene-type filter, relationship closeness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prompt_construction.retrieval.example_selector import (
    select_examples,
    _text_dedup,
    _filter_scene_types,
    _normalize_text_for_dedup,
    _relationship_closeness_score,
    _compute_rank_score,
    EXCLUDED_SCENE_TYPES_DEFAULT,
)


def _make_record(**overrides) -> dict:
    base = {
        "processed_id": "test_1",
        "canonical_id": "test_1",
        "character": "Abigail",
        "dialogue_type": "general_dialogue",
        "relationship_gate": "heart_min_4",
        "lang": "zh",
        "text_display": "你在做什么呢？",
        "scene_type": "daily",
        "season": "any",
        "weather": "any",
        "route": "any",
        "required_flags": "",
        "_distance": 0.35,
        "fallback_reason": None,
        "heart_min": 4,
    }
    base.update(overrides)
    return base


# ===================================================================
# Test: Text dedup
# ===================================================================

class TestTextDedup:

    def test_exact_duplicate_removed(self):
        records = [
            _make_record(text_display="你好", processed_id="a", canonical_id="a"),
            _make_record(text_display="你好", processed_id="b", canonical_id="b"),
        ]
        result = _text_dedup(records)
        assert len(result) == 1

    def test_different_text_kept(self):
        records = [
            _make_record(text_display="你好", processed_id="a", canonical_id="a"),
            _make_record(text_display="再见", processed_id="b", canonical_id="b"),
        ]
        result = _text_dedup(records)
        assert len(result) == 2

    def test_near_duplicate_removed(self):
        """Very similar texts with minor differences should be deduped."""
        records = [
            _make_record(text_display="今天天气真好啊，我们出去走走吧。", processed_id="a", canonical_id="a"),
            _make_record(text_display="今天天气真好，我们出去走走吧。", processed_id="b", canonical_id="b"),
        ]
        result = _text_dedup(records)
        assert len(result) == 1

    def test_same_canonical_different_text_kept(self):
        """Same canonical_id but different text — canonical dedup handles this."""
        records = [
            _make_record(text_display="你好", processed_id="a:zh", canonical_id="a", lang="zh"),
            _make_record(text_display="Hello", processed_id="a:en", canonical_id="a", lang="en"),
        ]
        # text_dedup doesn't look at canonical_id, only text content
        result = _text_dedup(records)
        assert len(result) == 2

    def test_empty_text_skipped(self):
        records = [
            _make_record(text_display="", processed_id="a", canonical_id="a"),
            _make_record(text_display="你好", processed_id="b", canonical_id="b"),
        ]
        result = _text_dedup(records)
        assert len(result) == 1

    def test_whitespace_only_difference_removed(self):
        records = [
            _make_record(text_display="你好 世界", processed_id="a", canonical_id="a"),
            _make_record(text_display="你好世界", processed_id="b", canonical_id="b"),
        ]
        result = _text_dedup(records)
        assert len(result) == 1


# ===================================================================
# Test: Normalize text for dedup
# ===================================================================

class TestNormalizeTextForDedup:

    def test_case_insensitive(self):
        assert _normalize_text_for_dedup("Hello") == _normalize_text_for_dedup("hello")

    def test_whitespace_stripped(self):
        assert _normalize_text_for_dedup("Hello  World") == _normalize_text_for_dedup("HelloWorld")

    def test_punctuation_stripped(self):
        assert _normalize_text_for_dedup("你好！") == _normalize_text_for_dedup("你好")

    def test_empty_returns_empty(self):
        assert _normalize_text_for_dedup("") == ""
        assert _normalize_text_for_dedup(None) == ""


# ===================================================================
# Test: Scene-type filter
# ===================================================================

class TestSceneTypeFilter:

    def test_daily_always_passes(self):
        records = [_make_record(scene_type="daily")]
        result = _filter_scene_types(records)
        assert len(result) == 1

    def test_birthday_filtered_by_default(self):
        records = [_make_record(scene_type="birthday")]
        result = _filter_scene_types(records)
        assert len(result) == 0

    def test_festival_filtered_by_default(self):
        records = [_make_record(scene_type="festival")]
        result = _filter_scene_types(records)
        assert len(result) == 0

    def test_marriage_filtered_by_default(self):
        records = [_make_record(scene_type="marriage")]
        result = _filter_scene_types(records)
        assert len(result) == 0

    def test_special_filtered_by_default(self):
        records = [_make_record(scene_type="special")]
        result = _filter_scene_types(records)
        assert len(result) == 0

    def test_gift_filtered_by_default(self):
        records = [_make_record(scene_type="gift")]
        result = _filter_scene_types(records)
        assert len(result) == 0

    def test_birthday_allowed_when_is_birthday(self):
        records = [_make_record(scene_type="birthday")]
        result = _filter_scene_types(records, {"is_birthday": True})
        assert len(result) == 1

    def test_festival_allowed_when_is_festival(self):
        records = [_make_record(scene_type="festival")]
        result = _filter_scene_types(records, {"is_festival": True})
        assert len(result) == 1

    def test_marriage_allowed_when_spouse(self):
        records = [_make_record(scene_type="marriage")]
        result = _filter_scene_types(records, {"relationship": "spouse"})
        assert len(result) == 1

    def test_marriage_allowed_when_married(self):
        records = [_make_record(scene_type="marriage")]
        result = _filter_scene_types(records, {"relationship": "married"})
        assert len(result) == 1

    def test_marriage_blocked_for_friend(self):
        records = [_make_record(scene_type="marriage")]
        result = _filter_scene_types(records, {"relationship": "friend"})
        assert len(result) == 0

    def test_no_excluded_types_passes_all(self):
        records = [
            _make_record(scene_type="birthday"),
            _make_record(scene_type="daily"),
        ]
        result = _filter_scene_types(records, excluded_types=frozenset())
        assert len(result) == 2

    def test_weather_always_passes(self):
        records = [_make_record(scene_type="weather")]
        result = _filter_scene_types(records)
        assert len(result) == 1

    def test_seasonal_always_passes(self):
        records = [_make_record(scene_type="seasonal")]
        result = _filter_scene_types(records)
        assert len(result) == 1


# ===================================================================
# Test: Relationship closeness scoring
# ===================================================================

class TestRelationshipCloseness:

    def test_same_stage_gets_full_bonus(self):
        rec = _make_record(relationship_gate="heart_min_4")
        score = _relationship_closeness_score(rec, "friend")
        assert score > 0
        # friend=4, heart_min_4=4, gap=0 → full bonus

    def test_distant_stage_gets_lower_bonus(self):
        rec_close = _make_record(relationship_gate="heart_min_4")
        rec_far = _make_record(relationship_gate="heart_min_2")
        score_close = _relationship_closeness_score(rec_close, "friend")
        score_far = _relationship_closeness_score(rec_far, "friend")
        assert score_close > score_far

    def test_any_gate_gets_zero(self):
        rec = _make_record(relationship_gate="any")
        score = _relationship_closeness_score(rec, "friend")
        assert score == 0.0

    def test_empty_relationship_gets_zero(self):
        rec = _make_record(relationship_gate="heart_min_4")
        score = _relationship_closeness_score(rec, "")
        assert score == 0.0

    def test_stranger_stage_for_friend(self):
        """Stranger gate at friend stage → still gets some bonus but less than close gate."""
        rec = _make_record(relationship_gate="stranger")
        score = _relationship_closeness_score(rec, "friend")
        # stranger=0, friend=4, gap=4 → partial bonus
        assert 0 < score < 0.08

    def test_closeness_affects_rank_order(self):
        """Records with closer relationship gate should rank higher."""
        rec_close = _make_record(
            relationship_gate="heart_min_4",
            _distance=0.4,
            text_display="你好啊朋友",
        )
        rec_far = _make_record(
            relationship_gate="stranger",
            _distance=0.4,
            text_display="你好陌生人",
            canonical_id="test_far",
            processed_id="test_far",
        )
        score_close = _compute_rank_score(rec_close, "zh", "friend")
        score_far = _compute_rank_score(rec_far, "zh", "friend")
        assert score_close > score_far


# ===================================================================
# Test: select_examples integration with new features
# ===================================================================

class TestSelectExamplesIntegration:

    def test_scene_type_filter_in_select(self):
        """Birthday examples should be filtered out for normal friend query."""
        records = [
            _make_record(
                text_display="生日快乐！", scene_type="birthday",
                processed_id="bd_1", canonical_id="bd_1",
            ),
            _make_record(
                text_display="你在做什么呢？", scene_type="daily",
                processed_id="daily_1", canonical_id="daily_1",
            ),
            _make_record(
                text_display="今天天气不错。", scene_type="daily",
                processed_id="daily_2", canonical_id="daily_2",
                _distance=0.36,
            ),
        ]
        selected = select_examples(
            records, target_lang="zh", max_distance=1.0,
            current_state={"relationship": "friend"},
        )
        # Birthday should be filtered out
        for rec in selected:
            assert rec.get("scene_type") != "birthday"

    def test_scene_type_allowed_when_matched(self):
        """Birthday examples should pass when is_birthday=True."""
        records = [
            _make_record(
                text_display="生日快乐！", scene_type="birthday",
                processed_id="bd_1", canonical_id="bd_1",
            ),
            _make_record(
                text_display="你在做什么呢？", scene_type="daily",
                processed_id="daily_1", canonical_id="daily_1",
            ),
        ]
        selected = select_examples(
            records, target_lang="zh", max_distance=1.0,
            current_state={"relationship": "friend", "is_birthday": True},
        )
        # Birthday should be allowed
        scene_types = [r.get("scene_type") for r in selected]
        assert "birthday" in scene_types

    def test_text_dedup_in_select(self):
        """Duplicate text should be removed."""
        records = [
            _make_record(
                text_display="你好", processed_id="a", canonical_id="a",
                relationship_gate="any", _distance=0.3,
            ),
            _make_record(
                text_display="你好", processed_id="b", canonical_id="b",
                relationship_gate="heart_min_4", _distance=0.35,
            ),
            _make_record(
                text_display="再见", processed_id="c", canonical_id="c",
                relationship_gate="any", _distance=0.4,
            ),
        ]
        selected = select_examples(
            records, target_lang="zh", max_distance=1.0,
            current_state={"relationship": "friend"},
        )
        texts = [r.get("text_display") for r in selected]
        assert texts.count("你好") <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
