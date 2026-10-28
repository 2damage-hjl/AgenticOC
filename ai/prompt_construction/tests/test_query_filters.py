"""Tests for game-state filtering logic (retrieval/filters.py).

No heavy dependencies required — tests run without embedding model or LanceDB.

Covers:
- Route visibility
- Required flags visibility
- Language filtering with fallback
- Context field filtering
- Post-filter pipeline
- Marriage flag specificity
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

from prompt_construction.retrieval.filters import (
    filter_by_lang_fallback,
    post_filter,
    required_flags_visible,
    route_visible,
)


# ===================================================================
# Test: Route visibility
# ===================================================================

class TestRouteVisible:

    def test_route_any_always_visible(self):
        assert route_visible("any", "pre_choice") is True
        assert route_visible("any", "community_center_completed") is True
        assert route_visible("any", "joja_active") is True
        assert route_visible("any", "joja_completed") is True

    def test_route_any_empty_string(self):
        assert route_visible("", "pre_choice") is True

    def test_pre_choice_visible_in_pre_choice(self):
        assert route_visible("pre_choice", "pre_choice") is True

    def test_pre_choice_not_visible_in_community_center(self):
        assert route_visible("pre_choice", "community_center_completed") is False

    def test_community_center_visible_in_community_center(self):
        assert route_visible("community_center", "community_center_completed") is True

    def test_community_center_not_visible_in_pre_choice(self):
        assert route_visible("community_center", "pre_choice") is False

    def test_joja_visible_in_joja_active(self):
        assert route_visible("joja", "joja_active") is True

    def test_joja_visible_in_joja_completed(self):
        assert route_visible("joja", "joja_completed") is True

    def test_joja_not_visible_in_pre_choice(self):
        assert route_visible("joja", "pre_choice") is False

    def test_post_joja_visible_in_joja_completed(self):
        assert route_visible("post_joja", "joja_completed") is True

    def test_post_joja_not_visible_in_joja_active(self):
        assert route_visible("post_joja", "joja_active") is False

    def test_non_joja_visible_in_pre_choice(self):
        assert route_visible("non_joja", "pre_choice") is True

    def test_non_joja_visible_in_community_center(self):
        assert route_visible("non_joja", "community_center_completed") is True

    def test_non_joja_not_visible_in_joja_active(self):
        assert route_visible("non_joja", "joja_active") is False

    def test_missing_route_defaults_to_any(self):
        """Missing route (empty/None) is treated as 'any' → always visible."""
        assert route_visible("", "pre_choice") is True
        assert route_visible("", "joja_completed") is True


# ===================================================================
# Test: Required flags visibility
# ===================================================================

class TestRequiredFlagsVisible:

    def test_empty_flags_always_visible(self):
        assert required_flags_visible("", set()) is True

    def test_single_flag_satisfied(self):
        assert required_flags_visible(
            "relationship.married_to.Abigail",
            {"relationship.married_to.Abigail"},
        ) is True

    def test_single_flag_not_satisfied(self):
        assert required_flags_visible(
            "relationship.married_to.Abigail",
            set(),
        ) is False

    def test_unmarried_does_not_pass_married_flag(self):
        """Unmarried game_flags do not satisfy relationship.married_to.Abigail."""
        assert required_flags_visible(
            "relationship.married_to.Abigail",
            {"world.community_center.completed"},
        ) is False

    def test_married_to_sebastian_not_equal_to_married_to_abigail(self):
        """married_to.Sebastian does NOT satisfy married_to.Abigail."""
        assert required_flags_visible(
            "relationship.married_to.Abigail",
            {"relationship.married_to.Sebastian"},
        ) is False

    def test_multiple_flags_all_required(self):
        assert required_flags_visible(
            "relationship.married_to.Abigail|world.ginger_island.beach_resort.opened",
            {"relationship.married_to.Abigail", "world.ginger_island.beach_resort.opened"},
        ) is True

    def test_multiple_flags_partial_not_satisfied(self):
        assert required_flags_visible(
            "relationship.married_to.Abigail|world.ginger_island.beach_resort.opened",
            {"relationship.married_to.Abigail"},
        ) is False

    def test_resort_flag(self):
        assert required_flags_visible(
            "world.ginger_island.beach_resort.opened",
            {"world.ginger_island.beach_resort.opened"},
        ) is True

    def test_missing_required_flags_defaults_to_visible(self):
        """Records with no required_flags (empty string) are always visible."""
        assert required_flags_visible("", set()) is True
        assert required_flags_visible("", {"something_else"}) is True


# ===================================================================
# Test: Language filtering with fallback
# ===================================================================

class TestFilterByLangFallback:

    def test_filter_zh(self):
        records = [
            {"processed_id": "1", "lang": "en", "text_display": "Hello"},
            {"processed_id": "2", "lang": "zh", "text_display": "你好"},
            {"processed_id": "3", "lang": "ja", "text_display": "こんにちは"},
        ]
        result = filter_by_lang_fallback(records, "zh", min_results=3)
        assert any(r["lang"] == "zh" for r in result)

    def test_filter_en(self):
        records = [
            {"processed_id": "1", "lang": "en", "text_display": "Hello"},
            {"processed_id": "2", "lang": "zh", "text_display": "你好"},
        ]
        result = filter_by_lang_fallback(records, "en", min_results=3)
        assert any(r["lang"] == "en" for r in result)

    def test_none_returns_all(self):
        records = [
            {"processed_id": "1", "lang": "en"},
            {"processed_id": "2", "lang": "zh"},
        ]
        result = filter_by_lang_fallback(records, None)
        assert len(result) == 2

    def test_empty_results(self):
        result = filter_by_lang_fallback([], "en")
        assert result == []

    def test_fallback_en(self):
        """If target_lang has too few results, English records are added."""
        records = [
            {"processed_id": "1", "lang": "zh", "text_display": "你好"},
            {"processed_id": "2", "lang": "en", "text_display": "Hello"},
            {"processed_id": "3", "lang": "en", "text_display": "Hi"},
        ]
        result = filter_by_lang_fallback(records, "zh", min_results=3)
        langs = [r["lang"] for r in result]
        assert "zh" in langs
        assert "en" in langs
        # en fallback records should have fallback_reason
        fb = [r for r in result if r.get("fallback_reason") == "fallback_en"]
        assert len(fb) > 0

    def test_fallback_any(self):
        """If target_lang + en still too few, any-language records are added."""
        records = [
            {"processed_id": "1", "lang": "zh", "text_display": "你好"},
            {"processed_id": "2", "lang": "ja", "text_display": "こんにちは"},
        ]
        result = filter_by_lang_fallback(records, "zh", min_results=3)
        langs = [r["lang"] for r in result]
        assert "ja" in langs
        fb = [r for r in result if r.get("fallback_reason") == "fallback_any"]
        assert len(fb) > 0

    def test_no_fallback_when_enough(self):
        """No fallback when target_lang already meets min_results."""
        records = [
            {"processed_id": "1", "lang": "zh", "text_display": "你好"},
            {"processed_id": "2", "lang": "zh", "text_display": "早上好"},
            {"processed_id": "3", "lang": "zh", "text_display": "晚安"},
            {"processed_id": "4", "lang": "en", "text_display": "Hello"},
        ]
        result = filter_by_lang_fallback(records, "zh", min_results=3)
        assert len(result) == 3
        assert all(r["lang"] == "zh" for r in result)
        assert not any(r.get("fallback_reason") for r in result)


# ===================================================================
# Test: Post-filter pipeline
# ===================================================================

class TestPostFilter:

    def test_marriage_filtered_for_unmarried(self):
        """Unmarried game_flags do not pass marriage records."""
        records = [
            {
                "lang": "en",
                "route": "any",
                "required_flags": "relationship.married_to.Abigail",
                "dialogue_type": "marriage_dialogue",
            },
            {
                "lang": "en",
                "route": "any",
                "required_flags": "",
                "dialogue_type": "general_dialogue",
            },
        ]
        result = post_filter(records, game_flags=set(), route_state="pre_choice", target_lang="en")
        assert len(result) == 1
        assert result[0]["dialogue_type"] == "general_dialogue"

    def test_marriage_visible_for_married(self):
        records = [
            {
                "lang": "en",
                "route": "any",
                "required_flags": "relationship.married_to.Abigail",
            },
        ]
        result = post_filter(
            records,
            game_flags={"relationship.married_to.Abigail"},
            route_state="community_center_completed",
            target_lang="en",
        )
        assert len(result) == 1

    def test_married_sebastian_not_visible_for_abigail(self):
        """married_to.Sebastian flag does not satisfy married_to.Abigail."""
        records = [
            {
                "lang": "en",
                "route": "any",
                "required_flags": "relationship.married_to.Abigail",
            },
        ]
        result = post_filter(
            records,
            game_flags={"relationship.married_to.Sebastian"},
            route_state="community_center_completed",
            target_lang="en",
        )
        assert len(result) == 0

    def test_lang_filter_applied(self):
        records = [
            {"lang": "en", "route": "any", "required_flags": ""},
            {"lang": "zh", "route": "any", "required_flags": ""},
        ]
        result = post_filter(records, game_flags=set(), route_state="pre_choice", target_lang="zh")
        assert len(result) == 1
        assert result[0]["lang"] == "zh"

    def test_route_filter_applied(self):
        records = [
            {"lang": "en", "route": "joja", "required_flags": ""},
            {"lang": "en", "route": "any", "required_flags": ""},
        ]
        result = post_filter(records, game_flags=set(), route_state="pre_choice")
        assert len(result) == 1
        assert result[0]["route"] == "any"

    def test_missing_route_defaults_to_any(self):
        """Missing route is treated as 'any' → always visible."""
        records = [
            {"lang": "en", "required_flags": ""},  # no route key
        ]
        result = post_filter(records, game_flags=set(), route_state="pre_choice")
        assert len(result) == 1

    def test_missing_required_flags_defaults_to_visible(self):
        """Missing required_flags → always visible."""
        records = [
            {"lang": "en", "route": "any"},  # no required_flags key
        ]
        result = post_filter(records, game_flags=set(), route_state="pre_choice")
        assert len(result) == 1

    def test_resort_flag_filtered(self):
        """Resort dialogue requires the resort flag."""
        records = [
            {
                "lang": "en",
                "route": "any",
                "required_flags": "world.ginger_island.beach_resort.opened",
            },
        ]
        # Without the flag
        result = post_filter(records, game_flags=set(), route_state="community_center_completed")
        assert len(result) == 0

        # With the flag
        result = post_filter(
            records,
            game_flags={"world.ginger_island.beach_resort.opened"},
            route_state="community_center_completed",
        )
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
