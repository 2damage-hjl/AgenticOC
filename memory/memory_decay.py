"""
轻量记忆衰减。

每天对旧的 episodic_event 做轻量衰减，但保护重要事件：
- importance >= 0.85 (high_importance) 的事件跳过
- topic_tags 包含 relationship_turning_point / favorite_gift / conflict 的事件跳过
- status == "consolidated" 的事件跳过（narrative_arc 的源事件）
- 普通事件 importance *= 0.995
- importance < 0.2 且超过 14 天未访问 → status = "dormant"

注意：不做 hard delete，只做轻量 decay 和 dormant 标记。
"""
import json
from typing import List


DECAY_FACTOR = 0.995
DORMANT_THRESHOLD = 0.2
DORMANT_DAYS = 14

# 受保护 topic_tags：包含这些 tag 的事件不衰减
PROTECTED_TOPIC_TAGS = {
    "relationship_turning_point",
    "favorite_gift",
    "conflict",
}


def _safe_load_topic_tags(meta: dict) -> list:
    """从 metadata 中安全解析 topic_tags_json。"""
    raw = meta.get("topic_tags_json", "[]")
    if isinstance(raw, list):
        return raw
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _is_protected(meta: dict) -> bool:
    """判断事件是否受保护（不应衰减）。"""
    importance = float(meta.get("importance", 0.5))

    # 1. high_importance
    if importance >= 0.85:
        return True

    # 2. narrative_arc source event（已被周反思吸收）
    if meta.get("status") == "consolidated":
        return True

    # 3. topic_tags 保护
    tags = _safe_load_topic_tags(meta)
    if tags and set(tags) & PROTECTED_TOPIC_TAGS:
        return True

    # 4. 兼容旧式 boolean flag
    if meta.get("is_turning_point") in (True, "True", "true"):
        return True
    if meta.get("is_high_emotion") in (True, "True", "true"):
        return True
    if meta.get("is_favorite_gift") in (True, "True", "true"):
        return True

    return False


def apply_daily_decay(npc_id: str, current_day: int, store) -> None:
    """
    对指定 NPC 的旧 episodic_event 执行轻量衰减。
    """
    events = store.query_by_type(
        memory_type="episodic_event",
        npc_id=npc_id,
        include_archived=False,
    )

    if not events:
        return

    for event in events:
        meta = event.get("metadata", {})
        memory_id = event.get("memory_id", "")

        if not memory_id:
            continue

        # 跳过非 active
        status = meta.get("status", "active")
        if status != "active":
            continue

        # 保护重要事件
        if _is_protected(meta):
            continue

        # 衰减
        importance = float(meta.get("importance", 0.5))
        new_importance = importance * DECAY_FACTOR

        # 判断是否 dormant
        last_access = meta.get("last_access", meta.get("time", current_day))
        try:
            last_access = float(last_access)
        except (ValueError, TypeError):
            last_access = float(current_day)

        days_since_access = current_day - last_access

        new_status = status
        if new_importance < DORMANT_THRESHOLD and days_since_access > DORMANT_DAYS:
            new_status = "dormant"

        # 更新
        store.update_memory(
            memory_id,
            metadata={
                "importance": round(new_importance, 4),
                "status": new_status,
            },
            memory_type="episodic_event",
        )
