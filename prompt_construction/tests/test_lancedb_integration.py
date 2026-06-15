"""LanceDB integration tests — end-to-end against the real database.

These tests hit the actual LanceDB index on disk and require the BGE-M3
embedding model (cached locally).  They are **not** unit tests with mocked
data — they validate the full pipeline: query → filter → route → lang fallback.

Run:
    set TRANSFORMERS_OFFLINE=1
    python -m pytest prompt_construction/tests/test_lancedb_integration.py -v

Skip condition: if the LanceDB database or embedding model is unavailable,
the entire module is skipped gracefully (no CI failure due to missing assets).
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


# ---------------------------------------------------------------------------
# Skip condition: detect if real LanceDB + model are available
# ---------------------------------------------------------------------------

_DB_PATH = _PROJECT_ROOT / "prompt_construction" / "data" / "indexes" / "lancedb"
_DB_AVAILABLE = _DB_PATH.is_dir() and any(_DB_PATH.iterdir())
_MODEL_AVAILABLE = False

if _DB_AVAILABLE:
    try:
        from prompt_construction.retrieval.query_lancedb import get_embedding_model
        get_embedding_model()
        _MODEL_AVAILABLE = True
    except Exception:
        _MODEL_AVAILABLE = False

_SKIP_REASON = (
    "LanceDB database or BGE-M3 model not available locally. "
    "Set TRANSFORMERS_OFFLINE=1 and ensure the index has been built."
)

requires_lancedb = pytest.mark.skipif(
    not (_DB_AVAILABLE and _MODEL_AVAILABLE),
    reason=_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Imports (deferred — only needed when tests actually run)
# ---------------------------------------------------------------------------

import lancedb
from prompt_construction.retrieval.query_lancedb import search_dialogues
from prompt_construction.retrieval.filters import (
    filter_by_lang_fallback,
    post_filter,
    route_visible,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_table():
    """Open and return the LanceDB table."""
    db = lancedb.connect(str(_DB_PATH))
    return db.open_table("stardew_dialogues")


def _all_records_for_character(character: str) -> list[dict]:
    """Fetch all records for a character from LanceDB."""
    table = _db_table()
    return table.search().where(f'character = "{character}"', prefilter=True).limit(500).to_list()


# ===================================================================
# 1. Real LanceDB can find target NPC
# ===================================================================

@requires_lancedb
class TestRealSearchFindsTargetNPC:
    """search_dialogues with character=... should return that NPC's records."""

    def test_damon_found(self):
        results = search_dialogues(
            query_text="你好 Damon",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=5,
        )
        assert len(results) > 0, "Damon should have at least 1 result"
        for r in results:
            assert r["character"] == "Damon"

    def test_abigail_found(self):
        results = search_dialogues(
            query_text="今天天气怎么样",
            character="Abigail",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=5,
        )
        assert len(results) > 0, "Abigail should have at least 1 result"
        for r in results:
            assert r["character"] == "Abigail"

    def test_wrong_character_not_returned(self):
        """Searching for Damon should not return Abigail."""
        results = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=5,
        )
        characters = {r["character"] for r in results}
        assert "Abigail" not in characters


# ===================================================================
# 2. zh query prefers zh results
# ===================================================================

@requires_lancedb
class TestZhQueryPrefersZh:
    """When target_lang=zh, Chinese records should be prioritized."""

    def test_zh_records_preferred_over_en(self):
        results = search_dialogues(
            query_text="你好",
            character="Abigail",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=50,
        )
        zh_count = sum(1 for r in results if r.get("lang") == "zh")
        assert zh_count > 0, "Should have at least some zh results"

    def test_zh_results_have_no_fallback_reason(self):
        """zh records matched directly should NOT have fallback_reason."""
        results = search_dialogues(
            query_text="你好",
            character="Abigail",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=50,
        )
        zh_direct = [r for r in results if r.get("lang") == "zh" and not r.get("fallback_reason")]
        # Abigail has plenty of zh records — at least some should be direct
        assert len(zh_direct) > 0, "At least some zh results should be direct (no fallback)"


# ===================================================================
# 3. zh insufficient → fallback to en
# ===================================================================

@requires_lancedb
class TestZhFallbackToEn:
    """When zh results < min_lang_results, English records are added."""

    def test_fallback_en_when_zh_insufficient(self):
        """When min_lang_results > available zh records, English records are added."""
        # Use a character with both zh and en records
        results = search_dialogues(
            query_text="矿洞冒险",
            character="Abigail",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=30,
            search_k=100,
            min_lang_results=50,  # impossible to satisfy → force fallback
        )
        # Should have both zh and non-zh records
        langs = {r.get("lang") for r in results}
        assert "zh" in langs, "Should still have zh records"
        # With min_lang_results=50, there must be fallback records
        fb = [r for r in results if r.get("fallback_reason")]
        assert len(fb) > 0, "Should have fallback records when zh is insufficient"

    def test_fallback_en_has_reason(self):
        """English fallback records should carry fallback_reason='fallback_en'."""
        results = search_dialogues(
            query_text="你好",
            character="Abigail",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=50,
            min_lang_results=50,
        )
        en_fb = [r for r in results if r.get("fallback_reason") == "fallback_en"]
        assert len(en_fb) > 0, "Should have fallback_en records"


# ===================================================================
# 4. Damon records are not source=static
# ===================================================================

@requires_lancedb
class TestDamonNotStatic:
    """Damon's records should come from LanceDB (dynamic), not static fallback."""

    def test_damon_has_lancedb_records(self):
        """Damon should have records in LanceDB (not just static examples)."""
        results = search_dialogues(
            query_text="你好 Damon",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=50,
        )
        assert len(results) > 0, "Damon should have LanceDB records"
        # All results should have a canonical_id (from the DB)
        for r in results:
            assert r.get("canonical_id"), f"Record {r.get('processed_id')} should have canonical_id"

    def test_damon_records_have_proper_metadata(self):
        """Damon's LanceDB records should have proper season/weather/heart metadata."""
        records = _all_records_for_character("Damon")
        assert len(records) > 0, "Damon should have records in LanceDB"

        # Check that Damon has location-based records
        scenes = {r.get("scene", "any") for r in records}
        assert any(s != "any" for s in scenes), "Damon should have location-based records (Saloon, Beach, etc.)"

        # Check that Damon has heart-level gated records
        gates = {r.get("relationship_gate", "any") for r in records}
        assert any(g != "any" for g in gates), "Damon should have heart-level gated records"

    def test_damon_no_marriage_records(self):
        """Damon is non-dateable — should have no marriage_dialogue records."""
        records = _all_records_for_character("Damon")
        types = {r.get("dialogue_type") for r in records}
        assert "marriage_dialogue" not in types, "Damon should not have marriage_dialogue records"


# ===================================================================
# 5. Relationship gate filters real marriage_dialogue
# ===================================================================

@requires_lancedb
class TestRelationshipGateFiltersMarriage:
    """Marriage dialogue should be filtered out when game_flags don't include married flag."""

    def test_marriage_dialogue_filtered_without_flag(self):
        """Abigail's marriage_dialogue should be excluded when not married."""
        results_no_flag = search_dialogues(
            query_text="亲爱的早上好",
            character="Abigail",
            target_lang="zh",
            game_flags=set(),  # NOT married
            route_state="community_center_completed",
            top_k=20,
            search_k=100,
        )
        marriage_no_flag = [
            r for r in results_no_flag
            if r.get("dialogue_type") == "marriage_dialogue"
        ]
        assert len(marriage_no_flag) == 0, "Marriage dialogue should be filtered when not married"

    def test_marriage_dialogue_visible_with_flag(self):
        """Abigail's marriage_dialogue should be visible when married to Abigail."""
        results_with_flag = search_dialogues(
            query_text="亲爱的早上好",
            character="Abigail",
            target_lang="zh",
            game_flags={"relationship.married_to.Abigail"},
            route_state="community_center_completed",
            top_k=20,
            search_k=100,
        )
        marriage_with_flag = [
            r for r in results_with_flag
            if r.get("dialogue_type") == "marriage_dialogue"
        ]
        assert len(marriage_with_flag) > 0, "Marriage dialogue should be visible when married to Abigail"

    def test_married_to_sebastian_does_not_show_abigail_marriage(self):
        """Being married to Sebastian should NOT show Abigail's marriage dialogue."""
        results = search_dialogues(
            query_text="亲爱的早上好",
            character="Abigail",
            target_lang="zh",
            game_flags={"relationship.married_to.Sebastian"},
            route_state="community_center_completed",
            top_k=20,
            search_k=100,
        )
        marriage = [
            r for r in results
            if r.get("dialogue_type") == "marriage_dialogue"
        ]
        assert len(marriage) == 0, "married_to.Sebastian should not satisfy Abigail's marriage flag"


# ===================================================================
# 6. Route filter works with real data
# ===================================================================

@requires_lancedb
class TestRouteFilterWithRealData:
    """Route-tagged records should be correctly filtered by game route state."""

    def test_joja_records_not_visible_in_cc_route(self):
        """Records tagged route='joja' should not appear when route_state='community_center_completed'."""
        # Use a character known to have joja records (e.g. Pierre, Shane)
        results = search_dialogues(
            query_text="Joja公司工作",
            character="Shane",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=20,
            search_k=100,
        )
        joja_results = [r for r in results if r.get("route") == "joja"]
        assert len(joja_results) == 0, "joja records should be filtered in CC route"

    def test_joja_records_visible_in_joja_route(self):
        """Records tagged route='joja' should appear when route_state='joja_active'."""
        results = search_dialogues(
            query_text="Joja公司工作",
            character="Shane",
            target_lang="zh",
            route_state="joja_active",
            top_k=20,
            search_k=100,
        )
        joja_results = [r for r in results if r.get("route") == "joja"]
        assert len(joja_results) > 0, "joja records should be visible in joja route"

    def test_cc_records_not_visible_in_joja_route(self):
        """Records tagged route='community_center' should not appear in joja route."""
        # Lewis has cc_ records
        results = search_dialogues(
            query_text="社区中心",
            character="Lewis",
            target_lang="zh",
            route_state="joja_active",
            top_k=20,
            search_k=100,
        )
        cc_results = [r for r in results if r.get("route") == "community_center"]
        assert len(cc_results) == 0, "community_center records should be filtered in joja route"

    def test_cc_records_visible_in_cc_route(self):
        """Records tagged route='community_center' should appear in CC route."""
        results = search_dialogues(
            query_text="社区中心",
            character="Lewis",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=20,
            search_k=100,
        )
        cc_results = [r for r in results if r.get("route") == "community_center"]
        assert len(cc_results) > 0, "community_center records should be visible in CC route"

    def test_non_joja_records_not_visible_in_joja_completed(self):
        """non_joja records should NOT appear in joja_completed route."""
        results = search_dialogues(
            query_text="超市购物",
            character="Shane",
            target_lang="zh",
            route_state="joja_completed",
            top_k=20,
            search_k=100,
        )
        non_joja = [r for r in results if r.get("route") == "non_joja"]
        assert len(non_joja) == 0, "non_joja records should be filtered in joja_completed route"

    def test_non_cc_not_visible_in_cc_completed(self):
        """non_cc records should NOT appear in community_center_completed route."""
        results = search_dialogues(
            query_text="你好",
            character="Penny",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=20,
            search_k=100,
        )
        non_cc = [r for r in results if r.get("route") == "non_cc"]
        assert len(non_cc) == 0, "non_cc records should be filtered in CC route"

    def test_route_not_all_any(self):
        """Verify that route column is not entirely 'any' in the database."""
        table = _db_table()
        df = table.to_pandas()
        non_any = df[df["route"] != "any"]
        assert len(non_any) > 0, "There should be records with non-'any' route values"

    def test_route_visible_joja_vs_cc(self):
        """Pure filter unit test with real route groups."""
        assert not route_visible("joja", "community_center_completed")
        assert not route_visible("community_center", "joja_active")

    def test_route_visible_non_joja_vs_joja(self):
        """non_joja is visible in pre_choice and community_center, not in joja."""
        assert route_visible("non_joja", "pre_choice")
        assert route_visible("non_joja", "community_center_completed")
        assert not route_visible("non_joja", "joja_active")
        assert not route_visible("non_joja", "joja_completed")


# ===================================================================
# 7. Heart level filter works with real data
# ===================================================================

@requires_lancedb
class TestHeartLevelFilter:
    """heart_min threshold should exclude records above player's heart level."""

    def test_low_heart_excludes_high_heart_records(self):
        """With heart_level=2, records with heart_min=4+ should be excluded."""
        results = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            heart_level=2,
            top_k=20,
            search_k=100,
        )
        for r in results:
            hm = r.get("heart_min")
            if hm is not None:
                assert hm <= 2, f"heart_min={hm} should be excluded when heart_level=2"

    def test_high_heart_includes_all_records(self):
        """With heart_level=10, all heart-gated records should be included."""
        results_low = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            heart_level=2,
            top_k=20,
            search_k=100,
        )
        results_high = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            heart_level=10,
            top_k=20,
            search_k=100,
        )
        assert len(results_high) >= len(results_low), "Higher heart should return >= as many results"


# ===================================================================
# 8. Full few-shot pipeline integration
# ===================================================================

@requires_lancedb
class TestFewShotPipelineIntegration:
    """End-to-end: get_few_shot_examples against real LanceDB."""

    def test_damon_returns_dynamic_source(self):
        """Damon should get dynamic results from LanceDB (bypassing max_distance)."""
        from prompt_construction.retrieval.query_lancedb import search_dialogues
        from prompt_construction.retrieval.few_shot_provider import filter_by_relationship_gate

        # Step 1: verify LanceDB can find Damon records
        raw = search_dialogues(
            query_text="NPC: Damon\n玩家输入: 你好啊\n当前关系: friend",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            game_flags=set(),
            heart_level=4,
            relationship=None,
            top_k=10,
            search_k=40,
        )
        assert len(raw) > 0, "LanceDB should return Damon records"

        # Step 2: verify anti-spoiler gate filter works
        filtered = filter_by_relationship_gate(raw, "friend")
        assert len(filtered) > 0, "Some Damon records should pass friend gate"

        # Step 3: verify none are marriage_dialogue
        for r in filtered:
            assert r.get("dialogue_type") != "marriage_dialogue"

        # Step 4: verify Damon's records are NOT from static source
        # (they have canonical_id, distance, etc. — hallmarks of LanceDB)
        for r in filtered:
            assert r.get("canonical_id"), "LanceDB records should have canonical_id"
            assert "_distance" in r, "LanceDB records should have _distance"

    def test_abigail_marriage_gate_filter(self):
        """Abigail at 'friend' stage should NOT see marriage examples."""
        from prompt_construction.retrieval.few_shot_provider import get_few_shot_examples

        state = {
            "last_user_input": "你好",
            "npc_id": "Abigail",
            "relationship": "friend",
            "season": "winter",
            "weather": "rain",
        }
        npc_config = {
            "character_type": "native",
            "static_examples": "- [rain] 看着雨帘飘荡",
        }
        result = get_few_shot_examples("Abigail", state, npc_config)
        # The text should NOT contain marriage-related content
        if result.source in ("dynamic", "mixed"):
            assert "marriage" not in result.text.lower() or "married" not in result.text.lower()


# ===================================================================
# 9. max_distance=0.75 does not filter out all dynamic results
# ===================================================================

@requires_lancedb
class TestMaxDistanceFilter:
    """Verify that the production max_distance=0.75 threshold is reasonable."""

    def test_damon_dynamic_not_all_filtered(self):
        """max_distance=0.75 should NOT filter out all Damon dynamic results."""
        from prompt_construction.retrieval.query_lancedb import search_dialogues
        from prompt_construction.retrieval.few_shot_provider import filter_by_relationship_gate
        from prompt_construction.retrieval.example_selector import select_examples

        raw = search_dialogues(
            query_text="NPC: Damon\n玩家输入: 你好啊\n当前关系: friend",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            game_flags=set(),
            heart_level=4,
            relationship=None,
            top_k=10,
            search_k=40,
        )
        assert len(raw) > 0, "LanceDB should return Damon records"
        filtered = filter_by_relationship_gate(raw, "friend")
        selected = select_examples(
            filtered,
            target_lang="zh",
            min_output=3,
            max_output=5,
            max_distance=0.75,
        )
        assert len(selected) > 0, (
            "max_distance=0.75 should not filter out ALL Damon results. "
            f"Raw: {len(raw)}, filtered: {len(filtered)}, selected: {len(selected)}"
        )

    def test_abigail_dynamic_not_all_filtered(self):
        """max_distance=0.75 should NOT filter out all Abigail dynamic results."""
        from prompt_construction.retrieval.query_lancedb import search_dialogues
        from prompt_construction.retrieval.few_shot_provider import filter_by_relationship_gate
        from prompt_construction.retrieval.example_selector import select_examples

        raw = search_dialogues(
            query_text="NPC: Abigail\n玩家输入: 你好啊\n当前关系: friend",
            character="Abigail",
            target_lang="zh",
            route_state="community_center_completed",
            game_flags=set(),
            heart_level=4,
            relationship=None,
            top_k=10,
            search_k=40,
        )
        assert len(raw) > 0, "LanceDB should return Abigail records"
        filtered = filter_by_relationship_gate(raw, "friend")
        selected = select_examples(
            filtered,
            target_lang="zh",
            min_output=3,
            max_output=5,
            max_distance=0.75,
        )
        assert len(selected) > 0, (
            "max_distance=0.75 should not filter out ALL Abigail results. "
            f"Raw: {len(raw)}, filtered: {len(filtered)}, selected: {len(selected)}"
        )

    def test_max_distance_none_returns_all(self):
        """max_distance=None should return results regardless of distance."""
        from prompt_construction.retrieval.query_lancedb import search_dialogues
        from prompt_construction.retrieval.example_selector import select_examples

        raw = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=40,
        )
        if not raw:
            pytest.skip("No Damon records")
        selected = select_examples(
            raw, target_lang="zh", max_distance=None,
        )
        assert len(selected) > 0, "max_distance=None should not filter by distance"

    def test_max_distance_zero_filters_everything(self):
        """max_distance=0.0 should filter out all records (distance is always > 0)."""
        from prompt_construction.retrieval.query_lancedb import search_dialogues
        from prompt_construction.retrieval.example_selector import select_examples

        raw = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=40,
        )
        if not raw:
            pytest.skip("No Damon records")
        selected = select_examples(
            raw, target_lang="zh", max_distance=0.0,
        )
        assert len(selected) == 0, "max_distance=0.0 should filter out all records"

    def test_selected_records_have_similarity_field(self):
        """select_examples should annotate records with 'similarity' field."""
        from prompt_construction.retrieval.query_lancedb import search_dialogues
        from prompt_construction.retrieval.example_selector import select_examples

        raw = search_dialogues(
            query_text="你好",
            character="Damon",
            target_lang="zh",
            route_state="community_center_completed",
            top_k=10,
            search_k=40,
        )
        if not raw:
            pytest.skip("No Damon records")
        selected = select_examples(
            raw, target_lang="zh", max_distance=0.75,
        )
        for r in selected:
            assert "similarity" in r, "Selected records should have 'similarity' field"
            dist = float(r.get("_distance", 0.0))
            assert abs(r["similarity"] - (1.0 - dist)) < 0.01, (
                f"similarity should equal 1 - distance: "
                f"sim={r['similarity']}, dist={dist}"
            )

    def test_debug_distance_distribution(self):
        """get_few_shot_examples debug should include distance_distribution."""
        from prompt_construction.retrieval.few_shot_provider import get_few_shot_examples

        state = {
            "last_user_input": "你好",
            "npc_id": "Damon",
            "relationship": "friend",
            "season": "spring",
            "weather": "sun",
        }
        npc_config = {
            "character_type": "native",
            "static_examples": "- [sunny] 今天天气不错",
        }
        result = get_few_shot_examples("Damon", state, npc_config)
        dd = result.debug.get("distance_distribution", {})
        if dd:  # only if LanceDB was reached
            assert "min" in dd, "distance_distribution should have 'min'"
            assert "p50" in dd, "distance_distribution should have 'p50'"
            assert "p95" in dd, "distance_distribution should have 'p95'"
            assert dd["min"] <= dd["p50"] <= dd["p95"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
