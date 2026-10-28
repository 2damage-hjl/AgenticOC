"""
preference_belief 软更新逻辑。

规则：
- 没有旧 belief → 新建
- 旧 polarity 和新 polarity 一致 → confidence 上升，增加证据
- 旧 polarity 和新 polarity 冲突 → 不删除旧证据，confidence 下降，
  polarity 偏向最近事件但加 uncertain 前缀
- neutral / uncertain → 特殊处理
- 每个 belief_key 只保留一条当前 belief，通过 upsert 更新
"""
import json
from typing import Dict, List, Optional

# 允许的 polarity 值
VALID_POLARITIES = {
    "like", "dislike", "neutral", "uncertain",
    "uncertain_like", "uncertain_dislike", "uncertain_neutral",
}

# 冲突对（base polarity 级别）
_CONFLICT_PAIRS = {("like", "dislike"), ("dislike", "like")}

# evidence_event_ids 最大保留条数
MAX_EVIDENCE_IDS = 20


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


def _normalize_polarity(polarity: str) -> str:
    """
    规范化 polarity：
    - None / 空字符串 → uncertain
    - 统一转小写
    - 去掉前后空格
    - 不在允许列表中 → uncertain
    """
    if not polarity:
        return "uncertain"
    polarity = str(polarity).strip().lower()
    if polarity not in VALID_POLARITIES:
        return "uncertain"
    return polarity


def _base_polarity(polarity: str) -> str:
    """
    提取基础极性：
    - uncertain_like → like
    - uncertain_dislike → dislike
    - uncertain_neutral → neutral
    - uncertain → uncertain
    - like / dislike / neutral → 自身
    """
    if polarity.startswith("uncertain_"):
        suffix = polarity[len("uncertain_"):]
        if suffix in ("like", "dislike", "neutral"):
            return suffix
    return polarity


def _is_conflict(old_polarity: str, new_polarity: str) -> bool:
    """
    判断两个 polarity 是否强冲突。
    只有 like vs dislike / dislike vs like 算冲突。
    neutral / uncertain 不算强冲突。
    """
    old_base = _base_polarity(old_polarity)
    new_base = _base_polarity(new_polarity)
    return (old_base, new_base) in _CONFLICT_PAIRS


# =========================================================
#  核心函数
# =========================================================

def update_preference_belief(
    npc_id: str,
    target: str,
    topic: str,
    new_polarity: str,
    evidence_event_id: str,
    current_day: int,
    store,
    new_confidence: float = 0.7,
) -> dict:
    """
    更新或创建 NPC 对玩家偏好的主观判断 (preference_belief)。

    每个 belief_key 只维护一条当前 belief，通过 upsert 更新。
    不删除旧 episodic_event，不 hard delete，不 archive 旧 belief。

    Returns:
        更新后的 belief 字典 {"memory_id", "content", "metadata"}
    """
    # 规范化输入
    normalized_polarity = _normalize_polarity(new_polarity)
    new_base = _base_polarity(normalized_polarity)

    # 构造 belief_key 和 memory_id
    belief_key = f"{npc_id}.{target}.preference.{topic}"
    memory_id = f"belief_{npc_id}_{target}_preference_{topic}"

    # 查询旧 belief
    old_items = store.query_by_type(
        memory_type="preference_belief",
        npc_id=npc_id,
        where={"belief_key": belief_key},
        include_archived=False,
    )

    if len(old_items) > 1:
        print(f"[belief_update] WARNING: found {len(old_items)} beliefs for {belief_key}, using first")

    if not old_items:
        # 新建 belief
        return _create_belief(
            store=store,
            npc_id=npc_id,
            target=target,
            belief_key=belief_key,
            memory_id=memory_id,
            topic=topic,
            polarity=normalized_polarity,
            confidence=new_confidence,
            evidence_event_id=evidence_event_id,
            current_day=current_day,
        )
    else:
        # 更新已有 belief
        return _update_belief(
            store=store,
            old_belief=old_items[0],
            npc_id=npc_id,
            target=target,
            belief_key=belief_key,
            memory_id=memory_id,
            topic=topic,
            new_polarity=normalized_polarity,
            new_base=new_base,
            new_confidence=new_confidence,
            evidence_event_id=evidence_event_id,
            current_day=current_day,
        )


# =========================================================
#  新建 belief
# =========================================================

def _create_belief(
    store,
    npc_id: str,
    target: str,
    belief_key: str,
    memory_id: str,
    topic: str,
    polarity: str,
    confidence: float,
    evidence_event_id: str,
    current_day: int,
) -> dict:
    """创建一条新的 preference_belief。"""
    confidence = _clamp(confidence, 0.35, 0.9)
    evidence_event_ids = [evidence_event_id] if evidence_event_id else []

    content = (
        f"{npc_id} 目前认为 {target} 对 {topic} 的偏好是 {polarity}，"
        f"置信度为 {confidence:.2f}。"
    )

    metadata = {
        "memory_id": memory_id,
        "memory_type": "preference_belief",
        "npc_id": npc_id,
        "target": target,
        "belief_key": belief_key,
        "belief_type": "player_preference",
        "topic": topic,
        "polarity": polarity,
        "confidence": confidence,
        "evidence_event_ids_json": json.dumps(evidence_event_ids, ensure_ascii=False),
        "contradiction_count": 0,
        "time": current_day,
        "last_access": current_day,
        "importance": 0.75,
        "status": "active",
        "source": "dialogue",
    }

    store.add_memory(
        memory_type="preference_belief",
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
#  更新已有 belief
# =========================================================

def _update_belief(
    store,
    old_belief: dict,
    npc_id: str,
    target: str,
    belief_key: str,
    memory_id: str,
    topic: str,
    new_polarity: str,
    new_base: str,
    new_confidence: float,
    evidence_event_id: str,
    current_day: int,
) -> dict:
    """根据新证据更新已有的 preference_belief。"""
    old_meta = old_belief.get("metadata", {})
    old_polarity = old_meta.get("polarity", "uncertain")
    old_base = _base_polarity(old_polarity)
    old_confidence = float(old_meta.get("confidence", 0.5))
    old_contradiction = int(old_meta.get("contradiction_count", 0))

    # 解析旧 evidence_event_ids
    evidence_event_ids = _safe_load_json_list(
        old_meta.get("evidence_event_ids_json", "[]")
    )

    # 追加新 evidence_event_id（去重）
    if evidence_event_id and evidence_event_id not in evidence_event_ids:
        evidence_event_ids.append(evidence_event_id)

    # 最多保留最近 20 条
    evidence_event_ids = evidence_event_ids[-MAX_EVIDENCE_IDS:]

    # ---- 根据新旧 polarity 关系决定更新策略 ----

    if new_polarity == "uncertain":
        # 新证据不清楚：不改变 polarity，confidence 微降
        final_polarity = old_polarity
        final_confidence = _clamp(old_confidence * 0.95, 0.35, 0.95)
        contradiction_count = old_contradiction

    elif new_base == "neutral":
        # 新证据表明中性偏好
        final_polarity = "uncertain_neutral"
        final_confidence = _clamp(min(old_confidence, 0.6), 0.35, 0.95)
        contradiction_count = old_contradiction

    elif _is_conflict(old_polarity, new_polarity):
        # 冲突处理
        contradiction_count = old_contradiction + 1

        # 多次冲突 → 始终保持 uncertain 前缀
        if contradiction_count >= 2:
            final_polarity = f"uncertain_{new_base}"
        else:
            # 第一次冲突：偏向最近证据但加 uncertain 前缀
            final_polarity = f"uncertain_{new_base}"

        final_confidence = _clamp(
            old_confidence * 0.55 + new_confidence * 0.25,
            0.35, 0.75,
        )

    elif old_base == new_base:
        # 一致更新
        final_polarity = new_base  # uncertain_like + like → like
        final_confidence = _clamp(
            old_confidence + 0.15 * new_confidence,
            0.0, 0.95,
        )
        contradiction_count = old_contradiction

    else:
        # 其他非冲突不一致（如 like → neutral / uncertain → like 等）
        # 偏向最新，但不增强 confidence
        final_polarity = new_polarity
        final_confidence = _clamp(
            old_confidence * 0.7 + new_confidence * 0.3,
            0.35, 0.85,
        )
        contradiction_count = old_contradiction

    # 构建 content
    content = (
        f"{npc_id} 目前认为 {target} 对 {topic} 的偏好是 {final_polarity}，"
        f"置信度为 {final_confidence:.2f}。"
    )

    metadata = {
        "memory_id": memory_id,
        "memory_type": "preference_belief",
        "npc_id": npc_id,
        "target": target,
        "belief_key": belief_key,
        "belief_type": "player_preference",
        "topic": topic,
        "polarity": final_polarity,
        "confidence": final_confidence,
        "evidence_event_ids_json": json.dumps(evidence_event_ids, ensure_ascii=False),
        "contradiction_count": contradiction_count,
        "time": current_day,
        "last_access": current_day,
        "importance": 0.75,
        "status": "active",
        "source": "dialogue",
    }

    store.add_memory(
        memory_type="preference_belief",
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
#  事件适配函数
# =========================================================

def update_preference_belief_from_event(
    event: dict,
    state: dict,
    store,
) -> dict | None:
    """
    从一条 episodic_event 中提取 belief_update 信息，
    调用 update_preference_belief 核心函数。

    兼容 belief_update_json (Chroma metadata 格式) 和
    belief_update (纯 dict 格式)。

    如果 event 没有 belief_update 信息，返回 None。
    """
    metadata = event.get("metadata", {})

    # 读取 belief_update：优先 belief_update_json，兼容 belief_update
    belief_update = None
    belief_update_json = metadata.get("belief_update_json")
    if belief_update_json:
        belief_update = _safe_load_json_list(belief_update_json) if False else None
        try:
            belief_update = json.loads(belief_update_json)
        except (json.JSONDecodeError, TypeError):
            belief_update = None

    if belief_update is None:
        belief_update = metadata.get("belief_update")

    if not belief_update or not isinstance(belief_update, dict):
        return None

    belief_type = belief_update.get("belief_type", "")
    if belief_type != "player_preference":
        return None

    topic = belief_update.get("topic", "").strip()
    polarity = belief_update.get("polarity", "uncertain")
    confidence = float(belief_update.get("confidence", 0.5))

    if not topic:
        return None

    npc_id = state.get("npc_id", metadata.get("npc_id", "Damon"))
    current_day = state.get("time_num", metadata.get("time", 1))
    event_id = event.get("memory_id", "")

    return update_preference_belief(
        npc_id=npc_id,
        target="player",
        topic=topic,
        new_polarity=polarity,
        evidence_event_id=event_id,
        current_day=current_day,
        store=store,
        new_confidence=confidence,
    )
