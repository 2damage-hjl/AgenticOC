"""
relationship_impression 更新逻辑。

维护 NPC 对玩家的长期主观关系印象，包含五维：
- trust（信任）
- warmth（亲近 / 好感）
- familiarity（熟悉度）
- confusion（困惑感）
- resentment（怨气 / 反感）

所有数值范围 [0, 1]，根据 episodic_event 做轻量更新。
每个 NPC 对玩家只维护一条当前 impression，通过 upsert 更新。
"""
import json
from typing import Dict, List

# 默认初始值
DEFAULT_IMPRESSION = {
    "trust": 0.5,
    "warmth": 0.5,
    "familiarity": 0.1,
    "confusion": 0.0,
    "resentment": 0.0,
}

# evidence_event_ids 最大保留条数
MAX_EVIDENCE_IDS = 30

# 单个事件每个维度变化上限
MAX_DELTA = 0.08

# topic_tags → 各维度变化系数（乘以 importance）
TAG_RULES = {
    "self_disclosure": {
        "trust": 0.02,
        "warmth": 0.0,
        "familiarity": 0.02,
        "confusion": 0.0,
        "resentment": 0.0,
    },
    "gift": {
        "trust": 0.0,
        "warmth": 0.015,
        "familiarity": 0.01,
        "confusion": 0.0,
        "resentment": 0.0,
    },
    "favorite_gift": {
        "trust": 0.01,
        "warmth": 0.035,
        "familiarity": 0.015,
        "confusion": 0.0,
        "resentment": 0.0,
    },
    "help": {
        "trust": 0.025,
        "warmth": 0.015,
        "familiarity": 0.0,
        "confusion": 0.0,
        "resentment": 0.0,
    },
    "preference": {
        "trust": 0.0,
        "warmth": 0.0,
        "familiarity": 0.015,
        "confusion": 0.0,
        "resentment": 0.0,
    },
    "contradiction": {
        "trust": 0.0,
        "warmth": 0.0,
        "familiarity": 0.0,
        "confusion": 0.03,
        "resentment": 0.0,
    },
    "conflict": {
        "trust": -0.025,
        "warmth": -0.02,
        "familiarity": 0.0,
        "confusion": 0.0,
        "resentment": 0.035,
    },
    "insult": {
        "trust": -0.025,
        "warmth": -0.02,
        "familiarity": 0.0,
        "confusion": 0.0,
        "resentment": 0.035,
    },
    "apology": {
        "trust": 0.01,
        "warmth": 0.0,
        "familiarity": 0.0,
        "confusion": 0.0,
        "resentment": -0.025,
    },
}


# =========================================================
#  内部工具函数
# =========================================================

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """将数值限制在 [low, high] 范围内。"""
    return max(low, min(high, value))


def _safe_load_json_list(value) -> list:
    """安全解析 JSON 列表，失败时返回空列表。"""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        result = json.loads(value)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _safe_load_json_dict(value) -> dict:
    """安全解析 JSON 字典，失败时返回空字典。"""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        result = json.loads(value)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _get_event_id(event: dict) -> str:
    """从 event 中获取 memory_id。"""
    return event.get("memory_id") or event.get("metadata", {}).get("memory_id", "")


def _get_topic_tags(event: dict) -> list:
    """从 event metadata 中读取 topic_tags，兼容 topic_tags_json。"""
    meta = event.get("metadata", {})
    # 优先读 topic_tags (list)
    tags = meta.get("topic_tags")
    if tags and isinstance(tags, list):
        return tags
    # 兼容 topic_tags_json
    return _safe_load_json_list(meta.get("topic_tags_json"))


def _get_relationship_signal(event: dict) -> dict | None:
    """
    从 event metadata 中读取 relationship_signal。
    优先 relationship_signal_json，兼容 relationship_signal。
    返回 None 表示没有。
    """
    meta = event.get("metadata", {})

    # 优先 relationship_signal_json
    signal_json = meta.get("relationship_signal_json")
    if signal_json:
        signal = _safe_load_json_dict(signal_json)
        if signal:
            return signal

    # 兼容 relationship_signal (dict)
    signal = meta.get("relationship_signal")
    if isinstance(signal, dict) and signal:
        return signal

    return None


# =========================================================
#  核心函数
# =========================================================

def update_relationship_impression(
    npc_id: str,
    target: str,
    events: list[dict],
    current_day: int,
    store,
) -> dict:
    """
    根据当天 episodic_event 更新 NPC 对玩家的长期关系印象。

    每个 NPC 对每个 target 只维护一条当前 impression，通过 upsert 更新。

    Returns:
        更新后的 impression 字典 {"memory_id", "content", "metadata"}
    """
    impression_key = f"{npc_id}.{target}.relationship"
    memory_id = f"rel_{npc_id}_{target}"

    # 查询旧 impression
    old_items = store.query_by_type(
        memory_type="relationship_impression",
        npc_id=npc_id,
        where={"impression_key": impression_key},
        include_archived=False,
    )

    if len(old_items) > 1:
        print(f"[impression_update] WARNING: found {len(old_items)} impressions for {impression_key}, using first")

    if old_items:
        old_meta = old_items[0].get("metadata", {})
        trust = float(old_meta.get("trust", DEFAULT_IMPRESSION["trust"]))
        warmth = float(old_meta.get("warmth", DEFAULT_IMPRESSION["warmth"]))
        familiarity = float(old_meta.get("familiarity", DEFAULT_IMPRESSION["familiarity"]))
        confusion = float(old_meta.get("confusion", DEFAULT_IMPRESSION["confusion"]))
        resentment = float(old_meta.get("resentment", DEFAULT_IMPRESSION["resentment"]))
        evidence_event_ids = _safe_load_json_list(
            old_meta.get("evidence_event_ids_json", "[]")
        )
    else:
        trust = DEFAULT_IMPRESSION["trust"]
        warmth = DEFAULT_IMPRESSION["warmth"]
        familiarity = DEFAULT_IMPRESSION["familiarity"]
        confusion = DEFAULT_IMPRESSION["confusion"]
        resentment = DEFAULT_IMPRESSION["resentment"]
        evidence_event_ids = []

    # 根据 events 做轻量更新
    for event in events:
        meta = event.get("metadata", {})
        importance = float(meta.get("importance", 0.5))
        event_id = _get_event_id(event)

        # 追加 evidence_event_id（去重）
        if event_id and event_id not in evidence_event_ids:
            evidence_event_ids.append(event_id)

        # 优先读取 relationship_signal
        signal = _get_relationship_signal(event)
        if signal:
            # 使用 relationship_signal，不叠加 topic_tags 推断
            for dim in ("trust", "warmth", "familiarity", "confusion", "resentment"):
                if dim in signal:
                    delta = _clamp(float(signal[dim]), -MAX_DELTA, MAX_DELTA)
                    if dim == "trust":
                        trust += delta
                    elif dim == "warmth":
                        warmth += delta
                    elif dim == "familiarity":
                        familiarity += delta
                    elif dim == "confusion":
                        confusion += delta
                    elif dim == "resentment":
                        resentment += delta
        else:
            # 没有 relationship_signal，用 topic_tags 推断
            tags = _get_topic_tags(event)
            for tag in tags:
                rule = TAG_RULES.get(tag)
                if not rule:
                    continue
                trust += rule["trust"] * importance
                warmth += rule["warmth"] * importance
                familiarity += rule["familiarity"] * importance
                confusion += rule["confusion"] * importance
                resentment += rule["resentment"] * importance

    # 限制所有数值在 [0, 1]
    trust = _clamp(trust)
    warmth = _clamp(warmth)
    familiarity = _clamp(familiarity)
    confusion = _clamp(confusion)
    resentment = _clamp(resentment)

    # 最多保留最近 30 条 evidence
    evidence_event_ids = evidence_event_ids[-MAX_EVIDENCE_IDS:]

    # 生成自然语言 content
    content = (
        f"{npc_id} 对 {target} 的长期印象："
        f"信任 {trust:.2f}，亲近 {warmth:.2f}，熟悉 {familiarity:.2f}，"
        f"困惑 {confusion:.2f}，怨气 {resentment:.2f}。"
    )

    metadata = {
        "memory_id": memory_id,
        "memory_type": "relationship_impression",
        "npc_id": npc_id,
        "target": target,
        "impression_key": impression_key,
        "trust": trust,
        "warmth": warmth,
        "familiarity": familiarity,
        "confusion": confusion,
        "resentment": resentment,
        "evidence_event_ids_json": json.dumps(evidence_event_ids, ensure_ascii=False),
        "time": current_day,
        "last_access": current_day,
        "importance": 0.85,
        "status": "active",
    }

    store.add_memory(
        memory_type="relationship_impression",
        content=content,
        metadata=metadata,
        memory_id=memory_id,
        upsert=True,
    )

    return {
        "memory_id": memory_id,
        "content": content,
        "metadata": metadata,
    }


# =========================================================
#  END_DAY 适配函数
# =========================================================

def update_relationship_impression_from_events(
    npc_id: str,
    events: list[dict],
    state: dict,
    store,
    target: str = "player",
) -> dict:
    """
    END_DAY 管线适配入口。
    内部直接调用 update_relationship_impression。
    """
    return update_relationship_impression(
        npc_id=npc_id,
        target=target,
        events=events,
        current_day=state.get("time_num", 1),
        store=store,
    )
