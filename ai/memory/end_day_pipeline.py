"""
END_DAY 长期记忆结转管线。

执行顺序：
1. 读取 MidTermMemory
2. 筛选值得保存的事件
3. 写入 episodic_event
4. 更新 preference_belief
5. 更新 relationship_impression
6. 执行轻量 decay
7. 清空 MidTermMemory
"""
import json
import uuid
from typing import Dict, List


# ---------------------------------------------------------------------------
# 2. filter_episodic_events
# ---------------------------------------------------------------------------

def filter_episodic_events(candidates: List[Dict]) -> List[Dict]:
    """
    从 MidTermMemory.upgrade() 的结果中筛选值得进入长期记忆的事件。
    
    保留条件：
    - content 非空
    - importance >= 0.35
    
    过滤掉：
    - 空内容、importance 太低、普通寒暄
    """
    results = []
    for mem in candidates:
        # 只处理 episodic_event 类型（upgrade 可能混入 preference_belief）
        if mem.get("memory_type") != "episodic_event":
            continue

        content = mem.get("content", "").strip()
        importance = float(mem.get("importance", 0.5))

        if not content:
            continue

        if importance < 0.35:
            continue

        results.append(mem)

    return results


# ---------------------------------------------------------------------------
# 3. save_episodic_events
# ---------------------------------------------------------------------------

def save_episodic_events(
    event_memories: List[Dict],
    state: Dict,
    store,
) -> List[Dict]:
    """
    将筛选后的事件写入长期记忆库（episodic_event collection）。
    
    返回统一格式的 saved_events 列表，供后续 belief / impression 更新使用。
    """
    npc_id = state.get("npc_id", "Damon")
    current_day = state.get("time_num", 1)

    saved_events = []
    for mem in event_memories:
        memory_id = mem.get("memory_id", f"evt_{uuid.uuid4().hex[:8]}")

        # Chroma metadata 不支持 list/dict，转 JSON 字符串
        topic_tags = mem.get("topic_tags", [])
        belief_update = mem.get("belief_update", None)

        metadata = {
            "memory_id": memory_id,
            "npc_id": mem.get("npc_id", npc_id),
            "time": mem.get("time", current_day),
            "last_access": mem.get("time", current_day),
            "location": mem.get("location", state.get("location", "Unknown")),
            "importance": mem.get("importance", 0.5),
            "memory_type": "episodic_event",
            "status": "active",
            "source": mem.get("source", "dialogue"),
            "topic_tags_json": json.dumps(topic_tags, ensure_ascii=False) if topic_tags else "[]",
            "emotional_valence": mem.get("emotional_valence", 0.0),
        }

        # belief_update 单独存储为 JSON 字符串（如果存在）
        if belief_update is not None:
            metadata["belief_update_json"] = json.dumps(belief_update, ensure_ascii=False)

        store.add_memory(
            memory_type="episodic_event",
            content=mem["content"],
            metadata=metadata,
            memory_id=memory_id,
        )

        saved_events.append({
            "memory_id": memory_id,
            "content": mem["content"],
            "metadata": metadata,
        })

    return saved_events


# ---------------------------------------------------------------------------
# 4. update_daily_preference_beliefs  (管线入口，调用 belief_update)
# ---------------------------------------------------------------------------

def update_daily_preference_beliefs(
    saved_events: List[Dict],
    state: Dict,
    store,
) -> None:
    """
    根据当日保存的 episodic_event，更新 NPC 对玩家偏好的主观判断。
    """
    from memory.belief_update import update_preference_belief_from_event

    for event in saved_events:
        update_preference_belief_from_event(event, state, store)


# ---------------------------------------------------------------------------
# 5. update_daily_relationship_impression  (管线入口，调用 impression_update)
# ---------------------------------------------------------------------------

def update_daily_relationship_impression(
    saved_events: List[Dict],
    state: Dict,
    store,
) -> None:
    """
    根据当日保存的 episodic_event，更新 NPC 对玩家的长期关系印象。
    """
    from memory.impression_update import update_relationship_impression_from_events

    npc_id = state.get("npc_id", "Damon")
    update_relationship_impression_from_events(npc_id, saved_events, state, store)


# ---------------------------------------------------------------------------
# 6. apply_daily_memory_decay  (管线入口，调用 memory_decay)
# ---------------------------------------------------------------------------

def apply_daily_memory_decay(npc_id: str, current_day: int, store) -> None:
    """
    对旧的 episodic_event 做轻量衰减。
    """
    from memory.memory_decay import apply_daily_decay

    apply_daily_decay(npc_id, current_day, store)
