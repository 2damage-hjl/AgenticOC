"""
END_DAY 长期记忆结转管线验收测试。

运行方式：
    cd d:/DamonAI/ai
    python -m memory.tests.test_end_day_pipeline
"""
import os
import sys
import tempfile
import shutil
import json

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _fresh_store():
    """创建一个使用临时目录的干净 MemoryStore，避免污染正式数据。"""
    from memory.embedded import MemoryStore
    tmp = tempfile.mkdtemp(prefix="test_end_day_")
    store = MemoryStore(db_path=tmp)
    return store, tmp


def _cleanup(store, tmp):
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass


def _make_state(npc_id="Damon", time_num=5, location="Town"):
    return {
        "command": "END_DAY",
        "npc_id": npc_id,
        "time_num": time_num,
        "location": location,
    }


def _write_mid_term_file(npc_id, scenes, tmp_dir):
    """写入中期记忆文件到临时目录。"""
    # 需要修改 MidTermMemory 的文件路径
    from memory.MTMemory import MidTermMemory
    file_path = os.path.join(tmp_dir, f"mid_term_{npc_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    # 猴子补丁：让 MidTermMemory 从临时目录读取
    MidTermMemory.FILE_PATH_TEMPLATE = os.path.join(tmp_dir, "mid_term_{npc_id}.json")
    return file_path


# ---------------------------------------------------------------------------
# 测试 1：END_DAY 无中期记忆
# ---------------------------------------------------------------------------

def test_no_mid_memories():
    """当 MidTermMemory.load() 返回空列表时不报错、不写数据。"""
    from memory.end_day_pipeline import (
        filter_episodic_events,
        save_episodic_events,
        update_daily_preference_beliefs,
        update_daily_relationship_impression,
        apply_daily_memory_decay,
    )

    store, tmp = _fresh_store()
    try:
        state = _make_state()

        # 直接调用管线函数，不走 MidTermMemory.load
        candidates = []  # upgrade 返回空
        event_memories = filter_episodic_events(candidates)
        assert event_memories == [], "无中期记忆时 event_memories 应为空"

        saved_events = save_episodic_events(event_memories, state, store)
        assert saved_events == [], "无事件时 saved_events 应为空"

        update_daily_preference_beliefs(saved_events, state, store)
        update_daily_relationship_impression(saved_events, state, store)
        apply_daily_memory_decay("Damon", 5, store)

        # 验证没有写入任何记忆
        all_events = store.query_by_type("episodic_event", npc_id="Damon", include_archived=True)
        assert len(all_events) == 0, "不应写入任何 episodic_event"

        print("✅ 测试 1 通过：END_DAY 无中期记忆时不报错、不写数据")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 2：普通中期记忆写入 episodic_event
# ---------------------------------------------------------------------------

def test_save_episodic_event():
    """构造一条中期记忆，验证写入 episodic_event 后的字段。"""
    from memory.end_day_pipeline import filter_episodic_events, save_episodic_events

    store, tmp = _fresh_store()
    try:
        state = _make_state()

        candidates = [{
            "memory_id": "evt_test_001",
            "npc_id": "Damon",
            "content": "玩家告诉 Damon 自己讨厌紫水晶。",
            "time": 5,
            "location": "Town",
            "importance": 0.7,
            "memory_type": "episodic_event",
            "topic_tags": ["preference", "amethyst"],
            "belief_update": {
                "belief_type": "player_preference",
                "topic": "amethyst",
                "polarity": "dislike",
                "confidence": 0.7,
            },
            "emotional_valence": -0.3,
        }]

        event_memories = filter_episodic_events(candidates)
        assert len(event_memories) == 1, "importance=0.7 应通过筛选"

        saved_events = save_episodic_events(event_memories, state, store)
        assert len(saved_events) == 1

        # 验证 Chroma 中的记录
        mem = store.get_memory("evt_test_001", memory_type="episodic_event")
        assert mem is not None, "episodic_event 应存在"
        assert mem["metadata"]["memory_type"] == "episodic_event"
        assert mem["metadata"]["status"] == "active"
        assert mem["metadata"]["last_access"] == 5
        assert mem["content"] == "玩家告诉 Damon 自己讨厌紫水晶。"

        print("✅ 测试 2 通过：普通中期记忆写入 episodic_event 字段正确")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 3：preference_belief 被创建
# ---------------------------------------------------------------------------

def test_preference_belief_created():
    """同测试 2 的事件，验证 preference_belief 被创建。"""
    from memory.end_day_pipeline import (
        filter_episodic_events,
        save_episodic_events,
        update_daily_preference_beliefs,
    )

    store, tmp = _fresh_store()
    try:
        state = _make_state()

        candidates = [{
            "memory_id": "evt_test_003",
            "npc_id": "Damon",
            "content": "玩家告诉 Damon 自己讨厌紫水晶。",
            "time": 5,
            "location": "Town",
            "importance": 0.7,
            "memory_type": "episodic_event",
            "topic_tags": ["preference", "amethyst"],
            "belief_update": {
                "belief_type": "player_preference",
                "topic": "amethyst",
                "polarity": "dislike",
                "confidence": 0.7,
            },
        }]

        saved_events = save_episodic_events(
            filter_episodic_events(candidates), state, store
        )
        update_daily_preference_beliefs(saved_events, state, store)

        # 查询 belief
        beliefs = store.query_by_type(
            memory_type="preference_belief",
            npc_id="Damon",
            where={"belief_key": "Damon.player.preference.amethyst"},
            include_archived=False,
        )
        assert len(beliefs) >= 1, "应该创建至少一条 preference_belief"

        belief = beliefs[0]
        assert belief["metadata"]["polarity"] == "dislike"
        assert float(belief["metadata"]["confidence"]) > 0

        evidence = json.loads(belief["metadata"]["evidence_event_ids_json"])
        assert "evt_test_003" in evidence, "evidence_event_ids 应包含当前 event_id"

        print("✅ 测试 3 通过：preference_belief 被正确创建")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 4：偏好冲突不会删除旧 event
# ---------------------------------------------------------------------------

def test_preference_conflict():
    """Day 1: 玩家喜欢紫水晶 → Day 5: 玩家讨厌紫水晶，冲突后不删旧 event。"""
    from memory.end_day_pipeline import (
        filter_episodic_events,
        save_episodic_events,
        update_daily_preference_beliefs,
    )

    store, tmp = _fresh_store()
    try:
        # Day 1: 喜欢
        state_day1 = _make_state(time_num=1)
        candidates_day1 = [{
            "memory_id": "evt_like_001",
            "npc_id": "Damon",
            "content": "玩家告诉 Damon 自己喜欢紫水晶。",
            "time": 1,
            "location": "Town",
            "importance": 0.7,
            "memory_type": "episodic_event",
            "topic_tags": ["preference", "amethyst"],
            "belief_update": {
                "belief_type": "player_preference",
                "topic": "amethyst",
                "polarity": "like",
                "confidence": 0.8,
            },
        }]
        saved1 = save_episodic_events(filter_episodic_events(candidates_day1), state_day1, store)
        update_daily_preference_beliefs(saved1, state_day1, store)

        # Day 5: 讨厌
        state_day5 = _make_state(time_num=5)
        candidates_day5 = [{
            "memory_id": "evt_dislike_005",
            "npc_id": "Damon",
            "content": "玩家告诉 Damon 自己讨厌紫水晶。",
            "time": 5,
            "location": "Town",
            "importance": 0.7,
            "memory_type": "episodic_event",
            "topic_tags": ["preference", "amethyst"],
            "belief_update": {
                "belief_type": "player_preference",
                "topic": "amethyst",
                "polarity": "dislike",
                "confidence": 0.7,
            },
        }]
        saved5 = save_episodic_events(filter_episodic_events(candidates_day5), state_day5, store)
        update_daily_preference_beliefs(saved5, state_day5, store)

        # 验证：episodic_event 中保留两条 event
        all_events = store.query_by_type("episodic_event", npc_id="Damon", include_archived=True)
        event_ids = [e["memory_id"] for e in all_events]
        assert "evt_like_001" in event_ids, "旧的 like event 不应被删除"
        assert "evt_dislike_005" in event_ids, "新的 dislike event 应存在"

        # 验证：preference_belief 中 polarity 变成 uncertain_dislike
        beliefs = store.query_by_type(
            memory_type="preference_belief",
            npc_id="Damon",
            where={"belief_key": "Damon.player.preference.amethyst"},
            include_archived=False,
        )
        assert len(beliefs) >= 1
        polarity = beliefs[0]["metadata"]["polarity"]
        assert "uncertain" in polarity or "dislike" in polarity, f"polarity 应为 uncertain_dislike，实际: {polarity}"
        
        confidence = float(beliefs[0]["metadata"]["confidence"])
        assert confidence < 0.95, f"冲突后 confidence 不应接近 1，实际: {confidence}"
        
        contradiction_count = int(beliefs[0]["metadata"].get("contradiction_count", 0))
        assert contradiction_count >= 1, "contradiction_count 应 >= 1"

        print("✅ 测试 4 通过：偏好冲突不会删除旧 event，polarity 变为 uncertain_dislike")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 5：relationship_impression 被更新
# ---------------------------------------------------------------------------

def test_relationship_impression_updated():
    """构造 gift / preference / contradiction 事件后，验证 relationship_impression。"""
    from memory.end_day_pipeline import (
        filter_episodic_events,
        save_episodic_events,
        update_daily_relationship_impression,
    )

    store, tmp = _fresh_store()
    try:
        state = _make_state()

        events_data = [
            {
                "memory_id": "evt_gift_001",
                "npc_id": "Damon",
                "content": "玩家送给 Damon 一束花。",
                "time": 5,
                "location": "Town",
                "importance": 0.8,
                "memory_type": "episodic_event",
                "topic_tags": ["gift", "flower"],
            },
            {
                "memory_id": "evt_pref_002",
                "npc_id": "Damon",
                "content": "玩家透露自己喜欢钓鱼。",
                "time": 5,
                "location": "Town",
                "importance": 0.6,
                "memory_type": "episodic_event",
                "topic_tags": ["preference", "fishing"],
            },
        ]

        saved = save_episodic_events(filter_episodic_events(events_data), state, store)
        update_daily_relationship_impression(saved, state, store)

        # 查询 impression
        impressions = store.query_by_type(
            memory_type="relationship_impression",
            npc_id="Damon",
            include_archived=False,
        )
        assert len(impressions) >= 1, "应该创建 relationship_impression"

        imp = impressions[0]
        meta = imp["metadata"]
        assert meta["memory_id"] == "rel_Damon_player"

        # familiarity 应上升（默认 0.1 → 更高）
        familiarity = float(meta["familiarity"])
        assert familiarity > 0.1, f"familiarity 应上升，实际: {familiarity}"

        # warmth 应因 gift 上升
        warmth = float(meta["warmth"])
        assert warmth > 0.5, f"warmth 应因 gift 上升，实际: {warmth}"

        print("✅ 测试 5 通过：relationship_impression 被正确更新")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 6：daily decay 生效
# ---------------------------------------------------------------------------

def test_daily_decay():
    """旧的 episodic_event 在 decay 后 importance 下降，超期变 dormant。"""
    from memory.end_day_pipeline import apply_daily_memory_decay

    store, tmp = _fresh_store()
    try:
        # 插入一条旧事件：importance=0.5, time=1, last_access=1
        store.add_memory(
            memory_type="episodic_event",
            content="一条普通的旧记忆。",
            metadata={
                "npc_id": "Damon",
                "time": 1,
                "last_access": 1,
                "location": "Town",
                "importance": 0.5,
                "status": "active",
            },
            memory_id="evt_old_decay_001",
        )

        # 插入一条重要事件（不应被 decay）
        store.add_memory(
            memory_type="episodic_event",
            content="一条非常重要的记忆。",
            metadata={
                "npc_id": "Damon",
                "time": 1,
                "last_access": 1,
                "location": "Town",
                "importance": 0.9,
                "status": "active",
            },
            memory_id="evt_important_002",
        )

        # 执行 decay（当前 day=21，距 last_access=1 已过 20 天）
        apply_daily_memory_decay("Damon", 21, store)

        # 验证旧事件：importance 下降
        old_mem = store.get_memory("evt_old_decay_001", memory_type="episodic_event")
        assert old_mem is not None
        new_importance = float(old_mem["metadata"]["importance"])
        assert new_importance < 0.5, f"importance 应下降，实际: {new_importance}"

        # importance < 0.2 且 > 14 天 → dormant
        # 0.5 * 0.995 = 0.4975，一次衰减不会到 0.2 以下
        # 所以这条不会变 dormant，只会 importance 下降
        assert old_mem["metadata"]["status"] == "active"

        # 验证重要事件：importance 不变
        imp_mem = store.get_memory("evt_important_002", memory_type="episodic_event")
        assert imp_mem is not None
        assert float(imp_mem["metadata"]["importance"]) == 0.9, "重要事件不应被 decay"

        print("✅ 测试 6a 通过：daily decay 使普通事件 importance 下降，保护重要事件")

        # --- 测试 dormant ---
        # 插入一条 importance=0.201 的旧事件
        store.add_memory(
            memory_type="episodic_event",
            content="快被遗忘的记忆。",
            metadata={
                "npc_id": "Damon",
                "time": 1,
                "last_access": 1,
                "location": "Town",
                "importance": 0.201,
                "status": "active",
            },
            memory_id="evt_dormant_003",
        )

        # decay: 0.201 * 0.995 = 0.199995 < 0.2, 且 21-1=20 > 14 → dormant
        apply_daily_memory_decay("Damon", 21, store)

        dormant_mem = store.get_memory("evt_dormant_003", memory_type="episodic_event")
        assert dormant_mem is not None
        assert dormant_mem["metadata"]["status"] == "dormant", \
            f"importance < 0.2 且超 14 天应变 dormant，实际: {dormant_mem['metadata']['status']}"

        # 不会 hard delete
        assert dormant_mem["content"] == "快被遗忘的记忆。"

        print("✅ 测试 6b 通过：超期低 importance 事件变 dormant，不会 hard delete")
    finally:
        _cleanup(store, tmp)


# ---------------------------------------------------------------------------
# 测试 7：filter_episodic_events 过滤低 importance 和空内容
# ---------------------------------------------------------------------------

def test_filter_episodic_events():
    """验证筛选逻辑。"""
    from memory.end_day_pipeline import filter_episodic_events

    candidates = [
        {"memory_id": "1", "content": "有内容的记忆", "importance": 0.7, "memory_type": "episodic_event"},
        {"memory_id": "2", "content": "", "importance": 0.7, "memory_type": "episodic_event"},  # 空内容
        {"memory_id": "3", "content": "低重要性", "importance": 0.2, "memory_type": "episodic_event"},  # 低 importance
        {"memory_id": "4", "content": "  ", "importance": 0.6, "memory_type": "episodic_event"},  # 纯空格
        {"memory_id": "5", "content": "偏好记忆", "importance": 0.8, "memory_type": "preference_belief"},  # 非 episodic
    ]

    filtered = filter_episodic_events(candidates)
    assert len(filtered) == 1, f"应只保留 1 条，实际: {len(filtered)}"
    assert filtered[0]["memory_id"] == "1"

    print("✅ 测试 7 通过：filter_episodic_events 正确过滤空内容和低 importance")


# ---------------------------------------------------------------------------
# 测试 8：无 belief_update 的事件不会创建 belief
# ---------------------------------------------------------------------------

def test_no_belief_without_belief_update():
    """没有 belief_update_json 的事件不应创建 preference_belief。"""
    from memory.end_day_pipeline import (
        filter_episodic_events,
        save_episodic_events,
        update_daily_preference_beliefs,
    )

    store, tmp = _fresh_store()
    try:
        state = _make_state()

        candidates = [{
            "memory_id": "evt_no_belief_001",
            "npc_id": "Damon",
            "content": "玩家和 Damon 聊了天气。",
            "time": 5,
            "location": "Town",
            "importance": 0.5,
            "memory_type": "episodic_event",
            # 没有 belief_update 和 topic_tags
        }]

        saved = save_episodic_events(filter_episodic_events(candidates), state, store)
        update_daily_preference_beliefs(saved, state, store)

        # 不应有 preference_belief
        beliefs = store.query_by_type("preference_belief", npc_id="Damon", include_archived=True)
        assert len(beliefs) == 0, "无 belief_update 时不应创建 preference_belief"

        print("✅ 测试 8 通过：无 belief_update 的事件不会创建 belief")
    finally:
        _cleanup(store, tmp)


if __name__ == "__main__":
    test_no_mid_memories()
    test_save_episodic_event()
    test_preference_belief_created()
    test_preference_conflict()
    test_relationship_impression_updated()
    test_daily_decay()
    test_filter_episodic_events()
    test_no_belief_without_belief_update()
    print("\n🎉 全部 END_DAY 管线验收测试通过！")
