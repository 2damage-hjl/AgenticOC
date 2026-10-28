"""
relationship_impression 更新逻辑的单元测试。

运行方式：
    cd d:/DamonAI/ai
    python -m memory.tests.test_impression_update
"""
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _fresh_store():
    from memory.embedded import MemoryStore
    tmp = tempfile.mkdtemp(prefix="test_impression_")
    store = MemoryStore(db_path=tmp)
    return store, tmp


def _cleanup(store, tmp):
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 测试 1：无旧 impression 时创建
# ---------------------------------------------------------------------------

def test_create_impression():
    from memory.impression_update import update_relationship_impression

    store, tmp = _fresh_store()
    try:
        events = [
            {
                "memory_id": "evt_001",
                "content": "玩家主动告诉 Damon 自己最近的真实想法。",
                "metadata": {
                    "memory_id": "evt_001",
                    "npc_id": "Damon",
                    "importance": 0.7,
                    "topic_tags_json": json.dumps(["self_disclosure"]),
                },
            }
        ]

        result = update_relationship_impression(
            npc_id="Damon",
            target="player",
            events=events,
            current_day=1,
            store=store,
        )

        assert result is not None
        meta = result["metadata"]

        # memory_id
        assert meta["memory_id"] == "rel_Damon_player", \
            f"memory_id 错误: {meta['memory_id']}"

        # trust > 0.5
        assert float(meta["trust"]) > 0.5, \
            f"trust 应 > 0.5，实际: {meta['trust']}"

        # familiarity > 0.1
        assert float(meta["familiarity"]) > 0.1, \
            f"familiarity 应 > 0.1，实际: {meta['familiarity']}"

        # evidence_event_ids 包含 evt_001
        evidence_ids = json.loads(meta["evidence_event_ids_json"])
        assert "evt_001" in evidence_ids, \
            f"evidence 应包含 evt_001，实际: {evidence_ids}"

        print("✅ 测试 1 通过：无旧 impression 时创建")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 2：送喜欢的礼物
# ---------------------------------------------------------------------------

def test_favorite_gift():
    from memory.impression_update import update_relationship_impression

    store, tmp = _fresh_store()
    try:
        # 先创建默认 impression
        events_initial = [
            {
                "memory_id": "evt_001",
                "content": "玩家和 Damon 打招呼。",
                "metadata": {
                    "memory_id": "evt_001",
                    "npc_id": "Damon",
                    "importance": 0.3,
                    "topic_tags_json": json.dumps(["casual"]),
                },
            }
        ]
        update_relationship_impression(
            npc_id="Damon", target="player",
            events=events_initial, current_day=1, store=store,
        )

        # 送喜欢的礼物
        events_gift = [
            {
                "memory_id": "evt_002",
                "content": "玩家送给 Damon 最喜欢的紫水晶。",
                "metadata": {
                    "memory_id": "evt_002",
                    "npc_id": "Damon",
                    "importance": 0.8,
                    "topic_tags_json": json.dumps(["favorite_gift"]),
                },
            }
        ]
        result = update_relationship_impression(
            npc_id="Damon", target="player",
            events=events_gift, current_day=2, store=store,
        )

        meta = result["metadata"]

        # warmth 上升
        assert float(meta["warmth"]) > 0.5, \
            f"warmth 应上升，实际: {meta['warmth']}"

        # trust 小幅上升
        assert float(meta["trust"]) > 0.5, \
            f"trust 应小幅上升，实际: {meta['trust']}"

        # familiarity 小幅上升
        assert float(meta["familiarity"]) > 0.1, \
            f"familiarity 应小幅上升，实际: {meta['familiarity']}"

        print("✅ 测试 2 通过：送喜欢的礼物")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 3：前后矛盾
# ---------------------------------------------------------------------------

def test_contradiction():
    from memory.impression_update import update_relationship_impression

    store, tmp = _fresh_store()
    try:
        events = [
            {
                "memory_id": "evt_003",
                "content": "玩家前后说法矛盾。",
                "metadata": {
                    "memory_id": "evt_003",
                    "npc_id": "Damon",
                    "importance": 0.6,
                    "topic_tags_json": json.dumps(["contradiction"]),
                },
            }
        ]

        result = update_relationship_impression(
            npc_id="Damon", target="player",
            events=events, current_day=1, store=store,
        )

        meta = result["metadata"]

        # confusion 上升
        assert float(meta["confusion"]) > 0.0, \
            f"confusion 应上升，实际: {meta['confusion']}"

        # trust / warmth 不要大幅变化（接近默认值）
        assert 0.45 <= float(meta["trust"]) <= 0.55, \
            f"trust 不应大幅变化，实际: {meta['trust']}"
        assert 0.45 <= float(meta["warmth"]) <= 0.55, \
            f"warmth 不应大幅变化，实际: {meta['warmth']}"

        print("✅ 测试 3 通过：前后矛盾")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 4：冒犯 NPC
# ---------------------------------------------------------------------------

def test_conflict():
    from memory.impression_update import update_relationship_impression

    store, tmp = _fresh_store()
    try:
        events = [
            {
                "memory_id": "evt_004",
                "content": "玩家冒犯了 Damon。",
                "metadata": {
                    "memory_id": "evt_004",
                    "npc_id": "Damon",
                    "importance": 0.7,
                    "topic_tags_json": json.dumps(["conflict"]),
                },
            }
        ]

        result = update_relationship_impression(
            npc_id="Damon", target="player",
            events=events, current_day=1, store=store,
        )

        meta = result["metadata"]

        # resentment 上升
        assert float(meta["resentment"]) > 0.0, \
            f"resentment 应上升，实际: {meta['resentment']}"

        # trust 下降
        assert float(meta["trust"]) < 0.5, \
            f"trust 应下降，实际: {meta['trust']}"

        # warmth 下降
        assert float(meta["warmth"]) < 0.5, \
            f"warmth 应下降，实际: {meta['warmth']}"

        # 所有值在 [0, 1]
        for dim in ("trust", "warmth", "familiarity", "confusion", "resentment"):
            assert 0.0 <= float(meta[dim]) <= 1.0, \
                f"{dim} 不在 [0,1]，实际: {meta[dim]}"

        print("✅ 测试 4 通过：冒犯 NPC")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 5：relationship_signal 优先生效
# ---------------------------------------------------------------------------

def test_relationship_signal_priority():
    from memory.impression_update import update_relationship_impression

    store, tmp = _fresh_store()
    try:
        # event 同时有 relationship_signal 和 topic_tags
        events = [
            {
                "memory_id": "evt_005",
                "content": "玩家送礼。",
                "metadata": {
                    "memory_id": "evt_005",
                    "npc_id": "Damon",
                    "importance": 0.7,
                    "topic_tags_json": json.dumps(["gift"]),
                    "relationship_signal_json": json.dumps({
                        "trust": 0.04,
                        "warmth": 0.02,
                        "confusion": 0.01,
                    }),
                },
            }
        ]

        # 先用 relationship_signal 跑一次
        result = update_relationship_impression(
            npc_id="Damon", target="player",
            events=events, current_day=1, store=store,
        )
        meta_signal = result["metadata"]

        # 记录结果
        trust_with_signal = float(meta_signal["trust"])
        warmth_with_signal = float(meta_signal["warmth"])

        # 验证 relationship_signal 生效（trust 上升 0.04 被限制在 MAX_DELTA 内）
        assert trust_with_signal > 0.5, \
            f"trust 应上升，实际: {trust_with_signal}"

        # 清空后只用 topic_tags 跑一次做对比
        store2, tmp2 = _fresh_store()
        try:
            events_tags_only = [
                {
                    "memory_id": "evt_005",
                    "content": "玩家送礼。",
                    "metadata": {
                        "memory_id": "evt_005",
                        "npc_id": "Damon",
                        "importance": 0.7,
                        "topic_tags_json": json.dumps(["gift"]),
                        # 没有 relationship_signal
                    },
                }
            ]
            result2 = update_relationship_impression(
                npc_id="Damon", target="player",
                events=events_tags_only, current_day=1, store=store2,
            )
            meta_tags = result2["metadata"]

            # 两种方式结果应该不同，证明 signal 优先且不叠加
            trust_tags_only = float(meta_tags["trust"])
            # gift tags 不会增加 trust，所以 trust 应接近默认 0.5
            assert trust_tags_only == 0.5, \
                f"gift tags 不应增加 trust，实际: {trust_tags_only}"

            # warmth 应该都上升，但数值不同
            warmth_tags_only = float(meta_tags["warmth"])
            assert warmth_with_signal != warmth_tags_only or trust_with_signal != trust_tags_only, \
                "relationship_signal 和 topic_tags 推断结果不应完全相同"

        finally:
            _cleanup(store2, tmp2)

        print("✅ 测试 5 通过：relationship_signal 优先生效，不叠加 topic_tags")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 6：_safe_load_json_list / _safe_load_json_dict
# ---------------------------------------------------------------------------

def test_json_utils():
    from memory.impression_update import _safe_load_json_list, _safe_load_json_dict

    # _safe_load_json_list
    assert _safe_load_json_list(None) == []
    assert _safe_load_json_list("") == []
    assert _safe_load_json_list("[]") == []
    assert _safe_load_json_list('["a", "b"]') == ["a", "b"]
    assert _safe_load_json_list(["x"]) == ["x"]
    assert _safe_load_json_list("invalid") == []

    # _safe_load_json_dict
    assert _safe_load_json_dict(None) == {}
    assert _safe_load_json_dict("") == {}
    assert _safe_load_json_dict("{}") == {}
    assert _safe_load_json_dict('{"a": 1}') == {"a": 1}
    assert _safe_load_json_dict({"b": 2}) == {"b": 2}
    assert _safe_load_json_dict("invalid") == {}

    print("✅ 测试 6 通过：JSON 工具函数")


# ---------------------------------------------------------------------------
# 测试 7：适配函数 update_relationship_impression_from_events
# ---------------------------------------------------------------------------

def test_from_events_adapter():
    from memory.impression_update import update_relationship_impression_from_events

    store, tmp = _fresh_store()
    try:
        events = [
            {
                "memory_id": "evt_007",
                "content": "玩家帮助 Damon。",
                "metadata": {
                    "memory_id": "evt_007",
                    "npc_id": "Damon",
                    "importance": 0.6,
                    "topic_tags_json": json.dumps(["help"]),
                },
            }
        ]
        state = {"npc_id": "Damon", "time_num": 5}

        result = update_relationship_impression_from_events(
            npc_id="Damon", events=events, state=state, store=store,
        )

        assert result is not None
        meta = result["metadata"]
        assert meta["memory_id"] == "rel_Damon_player"
        assert float(meta["trust"]) > 0.5, "help 应增加 trust"

        print("✅ 测试 7 通过：update_relationship_impression_from_events 适配函数")
    finally:
        _cleanup(store, tmp)


if __name__ == "__main__":
    test_create_impression()
    test_favorite_gift()
    test_contradiction()
    test_conflict()
    test_relationship_signal_priority()
    test_json_utils()
    test_from_events_adapter()
    print("\n🎉 全部 impression_update 测试通过！")
