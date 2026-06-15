"""
preference_belief 软更新逻辑的单元测试。

运行方式：
    cd d:/DamonAI/ai
    python -m memory.tests.test_belief_update
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _fresh_store():
    from memory.embedded import MemoryStore
    tmp = tempfile.mkdtemp(prefix="test_belief_")
    store = MemoryStore(db_path=tmp)
    return store, tmp


def _cleanup(store, tmp):
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 测试 1：新建 belief
# ---------------------------------------------------------------------------

def test_create_belief():
    from memory.belief_update import update_preference_belief

    store, tmp = _fresh_store()
    try:
        result = update_preference_belief(
            npc_id="Damon",
            target="player",
            topic="amethyst",
            new_polarity="like",
            evidence_event_id="evt_001",
            current_day=1,
            store=store,
        )

        assert result is not None
        meta = result["metadata"]

        # belief_key
        assert meta["belief_key"] == "Damon.player.preference.amethyst", \
            f"belief_key 错误: {meta['belief_key']}"

        # polarity
        assert meta["polarity"] == "like", \
            f"polarity 应为 like，实际: {meta['polarity']}"

        # confidence > 0.5
        assert float(meta["confidence"]) > 0.5, \
            f"confidence 应 > 0.5，实际: {meta['confidence']}"

        # evidence_event_ids 包含 evt_001
        import json
        evidence_ids = json.loads(meta["evidence_event_ids_json"])
        assert "evt_001" in evidence_ids, \
            f"evidence 应包含 evt_001，实际: {evidence_ids}"

        # contradiction_count == 0
        assert meta["contradiction_count"] == 0

        # memory_type
        assert meta["memory_type"] == "preference_belief"

        print("✅ 测试 1 通过：新建 belief")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 2：一致增强
# ---------------------------------------------------------------------------

def test_consistent_reinforce():
    from memory.belief_update import update_preference_belief

    store, tmp = _fresh_store()
    try:
        # Day 1: like
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_001",
            current_day=1, store=store,
        )

        # Day 2: like again
        result = update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_002",
            current_day=2, store=store,
        )

        meta = result["metadata"]

        # 仍然是 like
        assert meta["polarity"] == "like", \
            f"polarity 应为 like，实际: {meta['polarity']}"

        # confidence 上升
        assert float(meta["confidence"]) > 0.7, \
            f"confidence 应上升，实际: {meta['confidence']}"

        # evidence 包含两条
        import json
        evidence_ids = json.loads(meta["evidence_event_ids_json"])
        assert "evt_001" in evidence_ids and "evt_002" in evidence_ids, \
            f"evidence 应包含 evt_001 和 evt_002，实际: {evidence_ids}"

        # contradiction_count == 0
        assert meta["contradiction_count"] == 0

        print("✅ 测试 2 通过：一致增强")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 3：冲突软更新
# ---------------------------------------------------------------------------

def test_conflict_soft_update():
    from memory.belief_update import update_preference_belief

    store, tmp = _fresh_store()
    try:
        # Day 1: like
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_001",
            current_day=1, store=store,
        )
        # Day 2: like reinforce
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_002",
            current_day=2, store=store,
        )

        # Day 5: dislike — 冲突
        result = update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="dislike", evidence_event_id="evt_003",
            current_day=5, store=store, new_confidence=0.7,
        )

        meta = result["metadata"]

        # polarity == uncertain_dislike
        assert meta["polarity"] == "uncertain_dislike", \
            f"polarity 应为 uncertain_dislike，实际: {meta['polarity']}"

        # confidence 下降，不接近 1
        confidence = float(meta["confidence"])
        assert confidence < 0.8, \
            f"冲突后 confidence 不应接近 1，实际: {confidence}"

        # contradiction_count == 1
        assert meta["contradiction_count"] == 1, \
            f"contradiction_count 应为 1，实际: {meta['contradiction_count']}"

        # evidence 保留三条
        import json
        evidence_ids = json.loads(meta["evidence_event_ids_json"])
        assert "evt_001" in evidence_ids, "旧 evt_001 不应被删除"
        assert "evt_002" in evidence_ids, "旧 evt_002 不应被删除"
        assert "evt_003" in evidence_ids, "新 evt_003 应存在"

        # 旧 episodic_event 没有被删除（这里只检查 belief 层面）
        all_events = store.query_by_type("episodic_event", npc_id="Damon", include_archived=True)
        # 我们没有向 episodic_event 写入数据，所以这里应该为空
        # 重点是：belief 更新不应该删除任何 episodic_event
        assert len(all_events) == 0, "belief 更新不应写入或删除 episodic_event"

        print("✅ 测试 3 通过：冲突软更新")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 4：多次冲突
# ---------------------------------------------------------------------------

def test_multiple_conflicts():
    from memory.belief_update import update_preference_belief

    store, tmp = _fresh_store()
    try:
        # Day 1: like
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_001",
            current_day=1, store=store,
        )
        # Day 2: like reinforce
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_002",
            current_day=2, store=store,
        )
        # Day 5: dislike (conflict 1)
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="dislike", evidence_event_id="evt_003",
            current_day=5, store=store, new_confidence=0.7,
        )
        # Day 7: like (conflict 2)
        result = update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_004",
            current_day=7, store=store, new_confidence=0.6,
        )

        meta = result["metadata"]

        # polarity 应包含 uncertain
        assert "uncertain" in meta["polarity"], \
            f"多次冲突后 polarity 应含 uncertain，实际: {meta['polarity']}"

        # contradiction_count >= 2
        assert meta["contradiction_count"] >= 2, \
            f"contradiction_count 应 >= 2，实际: {meta['contradiction_count']}"

        # confidence 不应高
        confidence = float(meta["confidence"])
        assert confidence < 0.8, \
            f"多次冲突后 confidence 不应高，实际: {confidence}"

        # 不应该变成高置信度 like
        assert meta["polarity"] != "like", \
            f"多次冲突后不应恢复成确定的 like，实际: {meta['polarity']}"

        print("✅ 测试 4 通过：多次冲突")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 5：neutral 处理
# ---------------------------------------------------------------------------

def test_neutral_polarity():
    from memory.belief_update import update_preference_belief

    store, tmp = _fresh_store()
    try:
        # Day 1: like
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_001",
            current_day=1, store=store,
        )
        # Day 2: neutral
        result = update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="neutral", evidence_event_id="evt_002",
            current_day=2, store=store,
        )

        meta = result["metadata"]
        assert meta["polarity"] == "uncertain_neutral", \
            f"neutral 应变为 uncertain_neutral，实际: {meta['polarity']}"

        # contradiction_count 不增加
        assert meta["contradiction_count"] == 0, \
            f"neutral 不应增加 contradiction_count，实际: {meta['contradiction_count']}"

        print("✅ 测试 5 通过：neutral 处理")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 6：uncertain 处理
# ---------------------------------------------------------------------------

def test_uncertain_polarity():
    from memory.belief_update import update_preference_belief

    store, tmp = _fresh_store()
    try:
        # Day 1: like, confidence=0.8
        update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="like", evidence_event_id="evt_001",
            current_day=1, store=store, new_confidence=0.8,
        )
        # Day 2: uncertain
        result = update_preference_belief(
            npc_id="Damon", target="player", topic="amethyst",
            new_polarity="uncertain", evidence_event_id="evt_002",
            current_day=2, store=store,
        )

        meta = result["metadata"]

        # polarity 不变
        assert meta["polarity"] == "like", \
            f"uncertain 不应改变 polarity，实际: {meta['polarity']}"

        # confidence 微降
        confidence = float(meta["confidence"])
        assert confidence < 0.8, \
            f"uncertain 应使 confidence 微降，实际: {confidence}"

        print("✅ 测试 6 通过：uncertain 处理")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 7：update_preference_belief_from_event 适配函数
# ---------------------------------------------------------------------------

def test_from_event_adapter():
    import json
    from memory.belief_update import update_preference_belief_from_event

    store, tmp = _fresh_store()
    try:
        event = {
            "memory_id": "evt_from_event_001",
            "content": "玩家告诉 Damon 自己讨厌紫水晶。",
            "metadata": {
                "npc_id": "Damon",
                "belief_update_json": json.dumps({
                    "belief_type": "player_preference",
                    "topic": "amethyst",
                    "polarity": "dislike",
                    "confidence": 0.7,
                }),
            },
        }
        state = {"npc_id": "Damon", "time_num": 5}

        result = update_preference_belief_from_event(event, state, store)

        assert result is not None
        meta = result["metadata"]
        assert meta["polarity"] == "dislike"
        assert meta["topic"] == "amethyst"

        # 无 belief_update 的事件返回 None
        event_no_belief = {
            "memory_id": "evt_no_belief",
            "content": "闲聊天气。",
            "metadata": {},
        }
        result_none = update_preference_belief_from_event(event_no_belief, state, store)
        assert result_none is None, "无 belief_update 应返回 None"

        print("✅ 测试 7 通过：update_preference_belief_from_event 适配函数")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 8：_normalize_polarity 和 _base_polarity
# ---------------------------------------------------------------------------

def test_polarity_utils():
    from memory.belief_update import _normalize_polarity, _base_polarity

    # _normalize_polarity
    assert _normalize_polarity(None) == "uncertain"
    assert _normalize_polarity("") == "uncertain"
    assert _normalize_polarity("LIKE") == "like"
    assert _normalize_polarity(" dislike ") == "dislike"
    assert _normalize_polarity("unknown") == "uncertain"
    assert _normalize_polarity("uncertain_like") == "uncertain_like"

    # _base_polarity
    assert _base_polarity("uncertain_like") == "like"
    assert _base_polarity("uncertain_dislike") == "dislike"
    assert _base_polarity("uncertain_neutral") == "neutral"
    assert _base_polarity("uncertain") == "uncertain"
    assert _base_polarity("like") == "like"
    assert _base_polarity("dislike") == "dislike"
    assert _base_polarity("neutral") == "neutral"

    print("✅ 测试 8 通过：_normalize_polarity 和 _base_polarity")


if __name__ == "__main__":
    test_create_belief()
    test_consistent_reinforce()
    test_conflict_soft_update()
    test_multiple_conflicts()
    test_neutral_polarity()
    test_uncertain_polarity()
    test_from_event_adapter()
    test_polarity_utils()
    print("\n🎉 全部 belief_update 测试通过！")
