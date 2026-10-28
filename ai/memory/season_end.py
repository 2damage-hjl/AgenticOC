"""
SEASON_END 季节结束重型 consolidation。

职责：
1. 物理删除已 archived 超过一季（28天）的记忆
2. 重建 relationship_impression summary

状态流转（由每周衰减处理，不在此文件）：
  active → dormant → archived → 物理删除

星露谷每季 28 天，SEASON_END 只在季节结束时触发一次。
"""
import json
from typing import Dict, List

from memory.impression_update import (
    DEFAULT_IMPRESSION,
    TAG_RULES,
    MAX_DELTA,
    _clamp,
    _safe_load_json_list,
    _safe_load_json_dict,
)


# =========================
#  常量
# =========================

SEASON_LENGTH = 28
ARCHIVE_RETENTION_DAYS = 28  # archived 后至少保留 28 天才删除
BULK_QUERY_LIMIT = 10000


def _days_since(meta: dict, current_day: int) -> int:
    """计算距上次访问的天数。"""
    last_access = meta.get("last_access", meta.get("time", current_day))
    try:
        last_access = float(last_access)
    except (ValueError, TypeError):
        last_access = float(current_day)
    return max(0, int(current_day - last_access))


class SeasonConsolidator:
    """季节结束重型 consolidation 执行器。"""

    # --------------------------------------------------
    # 主入口
    # --------------------------------------------------

    @classmethod
    def run_season_consolidation(
        cls,
        npc_id: str,
        current_day: int,
        store,
        llm=None,
    ) -> dict:
        """
        对单个 NPC 执行季节结束重型 consolidation。

        Returns:
            {
                "npc_id": str,
                "hard_deleted": int,
                "rebuilt_impression": bool,
            }
        """
        print(f"[SeasonEnd] === 开始季节 consolidation for {npc_id} (Day {current_day}) ===")

        hard_deleted = cls._hard_delete_archived(npc_id, current_day, store)
        rebuilt_impression = cls._rebuild_relationship_impression(npc_id, current_day, store)

        result = {
            "npc_id": npc_id,
            "hard_deleted": hard_deleted,
            "rebuilt_impression": rebuilt_impression,
        }
        print(f"[SeasonEnd] {npc_id} 完成: {result}")
        return result

    # --------------------------------------------------
    # 工具：获取全部 NPC ID
    # --------------------------------------------------

    @staticmethod
    def get_all_npc_ids(store) -> List[str]:
        """从所有长期记忆 collection 中提取出全部 NPC ID。"""
        npc_ids = set()
        for mt in store.MEMORY_TYPES:
            items = store.query_by_type(
                memory_type=mt,
                top_k=BULK_QUERY_LIMIT,
                include_archived=True,
            )
            for item in items:
                nid = item.get("metadata", {}).get("npc_id")
                if nid:
                    npc_ids.add(nid)
        return list(npc_ids)

    # --------------------------------------------------
    # 1. 物理删除过期 archived
    # --------------------------------------------------

    @classmethod
    def _hard_delete_archived(
        cls,
        npc_id: str,
        current_day: int,
        store,
    ) -> int:
        """删除已 archived 超过一季（28天）的记忆。

        状态流转由每周衰减(active→dormant→archived)处理，
        本方法只负责对已归档一季以上的记忆执行物理删除。
        """
        events = store.query_by_type(
            memory_type="episodic_event",
            npc_id=npc_id,
            top_k=BULK_QUERY_LIMIT,
            include_archived=True,
            where={"status": "archived"},
        )

        deleted_count = 0
        for evt in events:
            meta = evt.get("metadata", {})
            memory_id = evt.get("memory_id", "")
            if not memory_id:
                continue

            days_since_archived = _days_since(meta, current_day)
            if days_since_archived < ARCHIVE_RETENTION_DAYS:
                continue

            try:
                store.delete_memory(
                    memory_id=memory_id,
                    memory_type="episodic_event",
                    hard=True,
                )
                deleted_count += 1
            except Exception as e:
                print(f"[SeasonEnd] hard delete failed for {memory_id}: {e}")

        print(f"[SeasonEnd] {npc_id}: {deleted_count} archived memories hard-deleted")
        return deleted_count

    # --------------------------------------------------
    # 2. 重建 relationship_impression summary
    # --------------------------------------------------

    @classmethod
    def _rebuild_relationship_impression(
        cls,
        npc_id: str,
        current_day: int,
        store,
    ) -> bool:
        """
        从本季所有 active episodic_event 重新计算 relationship_impression。
        纠正日常增量更新累积的漂移误差。
        """
        season_start = current_day - SEASON_LENGTH

        events = store.get_events_in_time_range(npc_id, season_start, current_day)
        active_events = [
            e for e in events
            if e.get("metadata", {}).get("status", "active") == "active"
        ]

        if not active_events:
            print(f"[SeasonEnd] {npc_id}: no active events this season, skip impression rebuild")
            return False

        trust = DEFAULT_IMPRESSION["trust"]
        warmth = DEFAULT_IMPRESSION["warmth"]
        familiarity = DEFAULT_IMPRESSION["familiarity"]
        confusion = DEFAULT_IMPRESSION["confusion"]
        resentment = DEFAULT_IMPRESSION["resentment"]
        evidence_event_ids = []

        for event in active_events:
            meta = event.get("metadata", {})
            importance = float(meta.get("importance", 0.5))
            event_id = event.get("memory_id") or meta.get("memory_id", "")

            if event_id:
                evidence_event_ids.append(event_id)

            signal = None
            signal_json = meta.get("relationship_signal_json")
            if signal_json:
                signal = _safe_load_json_dict(signal_json)
            if not signal:
                signal_raw = meta.get("relationship_signal")
                if isinstance(signal_raw, dict):
                    signal = signal_raw

            if signal:
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
                tags = _safe_load_json_list(meta.get("topic_tags_json", "[]"))
                for tag in tags:
                    rule = TAG_RULES.get(tag)
                    if not rule:
                        continue
                    trust += rule["trust"] * importance
                    warmth += rule["warmth"] * importance
                    familiarity += rule["familiarity"] * importance
                    confusion += rule["confusion"] * importance
                    resentment += rule["resentment"] * importance

        trust = _clamp(trust)
        warmth = _clamp(warmth)
        familiarity = _clamp(familiarity)
        confusion = _clamp(confusion)
        resentment = _clamp(resentment)

        evidence_event_ids = evidence_event_ids[-30:]

        target = "player"
        impression_key = f"{npc_id}.{target}.relationship"
        memory_id = f"rel_{npc_id}_{target}"

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

        print(f"[SeasonEnd] {npc_id}: rebuilt relationship_impression")
        return True
