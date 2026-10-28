"""
轻量记忆衰减，每周执行一次（仅在 current_day % 7 == 0 时生效）。

- active → 每周 importance *= DECAY_FACTOR
- active 且 imp < DORMANT_THRESHOLD 且 14天未访问 → dormant
- dormant → 每周 importance *= DECAY_FACTOR
- dormant 且 imp < ARCHIVE_THRESHOLD → archived（冻结，不再衰减）
- archived → 不衰减，等待季节结束时物理删除
- 受保护事件(high_importance / 特殊 tag) → 不衰减

不做 hard delete。
"""
import json
from typing import List


DECAY_FACTOR = 0.965   # 周衰减率 ≈ 0.995^7
DORMANT_THRESHOLD = 0.2
DORMANT_DAYS = 14
ARCHIVE_THRESHOLD = 0.15

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

    if importance >= 0.85:
        return True

    if meta.get("status") == "consolidated":
        return True

    tags = _safe_load_topic_tags(meta)
    if tags and set(tags) & PROTECTED_TOPIC_TAGS:
        return True

    # 兼容旧式 boolean flag
    for flag in ("is_turning_point", "is_high_emotion", "is_favorite_gift"):
        if meta.get(flag) in (True, "True", "true"):
            return True

    return False


def apply_daily_decay(npc_id: str, current_day: int, store) -> None:
    """
    对指定 NPC 的 episodic_event 执行周衰减。
    仅在 current_day % 7 == 0 时执行，减半 ChromaDB 写操作。
    """
    if current_day % 7 != 0:
        return

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

        # active 和 dormant 参与衰减；archived/consolidated 冻结
        status = meta.get("status", "active")
        if status not in ("active", "dormant"):
            continue

        if _is_protected(meta):
            continue

        importance = float(meta.get("importance", 0.5))
        new_importance = round(importance * DECAY_FACTOR, 4)

        last_access = meta.get("last_access", meta.get("time", current_day))
        try:
            last_access = float(last_access)
        except (ValueError, TypeError):
            last_access = float(current_day)

        days_since_access = current_day - last_access

        new_status = status
        update_fields = {"importance": new_importance, "status": new_status}

        if status == "active" and new_importance < DORMANT_THRESHOLD and days_since_access > DORMANT_DAYS:
            new_status = "dormant"
            update_fields["status"] = "dormant"
        elif status == "dormant" and new_importance < ARCHIVE_THRESHOLD:
            new_status = "archived"
            update_fields["status"] = "archived"
            update_fields["last_access"] = float(current_day)  # 记录归档时间

        store.update_memory(
            memory_id,
            metadata=update_fields,
            memory_type="episodic_event",
        )
