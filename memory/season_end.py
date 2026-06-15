"""
SEASON_END 季节结束重型 consolidation。

职责：
1. 读取全局长期记忆
2. 归档已被 narrative_arc 总结的低重要 event → dormant
3. 进一步归档长期无人访问的 dormant event → archived
4. 合并重复 narrative_arc
5. 重建 preference_belief confidence
6. 重建 relationship_impression summary

归档规则（两步）：
  Step 1: consolidated + importance < 0.3 + 14天未访问 → dormant
  Step 2: dormant + importance < 0.15 + 28天未访问 → archived

星露谷每季 28 天，SEASON_END 只在季节结束时触发一次。
"""
import json
import uuid
from typing import Dict, List, Optional, Tuple

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

SEASON_LENGTH = 28  # 星露谷一季 28 天

ARCHIVE_IMPORTANCE_THRESHOLD = 0.3
ARCHIVE_DORMANT_DAYS = 14
DEEP_ARCHIVE_IMPORTANCE_THRESHOLD = 0.15
DEEP_ARCHIVE_DORMANT_DAYS = 28
ARC_MERGE_SIMILARITY = 0.82
CONFIDENCE_REGRESSION_RATE = 0.1   # confidence 向 0.5 回归速率
CONTRADICTION_EXTRA_DECAY = 0.05   # 每次矛盾额外衰减
BULK_QUERY_LIMIT = 10000           # 批量查询上限


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
                "dormant_count": int,
                "archived_count": int,
                "merged_arcs": int,
                "rebuilt_beliefs": int,
                "rebuilt_impression": bool,
            }
        """
        print(f"[SeasonEnd] === 开始季节 consolidation for {npc_id} (Day {current_day}) ===")

        dormant = cls._consolidate_to_dormant(npc_id, current_day, store)
        archived = cls._consolidate_to_archived(npc_id, current_day, store)
        merged = cls._merge_duplicate_narrative_arcs(npc_id, current_day, store, llm)
        rebuilt_beliefs = cls._rebuild_belief_confidence(npc_id, current_day, store)
        rebuilt_impression = cls._rebuild_relationship_impression(npc_id, current_day, store)

        result = {
            "npc_id": npc_id,
            "dormant_count": dormant,
            "archived_count": archived,
            "merged_arcs": merged,
            "rebuilt_beliefs": rebuilt_beliefs,
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
    # 1. consolidated → dormant
    # --------------------------------------------------

    @classmethod
    def _consolidate_to_dormant(
        cls,
        npc_id: str,
        current_day: int,
        store,
    ) -> int:
        """
        已被 narrative_arc 总结 且 importance < 0.3 且 14天未访问 → dormant。
        """
        events = store.query_by_type(
            memory_type="episodic_event",
            npc_id=npc_id,
            top_k=BULK_QUERY_LIMIT,
            include_archived=False,
        )

        dormant_count = 0
        for evt in events:
            meta = evt.get("metadata", {})
            status = meta.get("status", "active")

            # 只处理 consolidated（已被 narrative_arc 总结）
            if status != "consolidated":
                continue

            importance = float(meta.get("importance", 0.5))
            if importance >= ARCHIVE_IMPORTANCE_THRESHOLD:
                continue

            # 检查 14 天未访问
            days_since_access = _days_since(meta, current_day)
            if days_since_access < ARCHIVE_DORMANT_DAYS:
                continue

            store.update_memory(
                memory_id=evt["memory_id"],
                memory_type="episodic_event",
                metadata={
                    "status": "dormant",
                    "last_access": current_day,
                },
            )
            dormant_count += 1

        print(f"[SeasonEnd] {npc_id}: {dormant_count} consolidated events → dormant")
        return dormant_count

    # --------------------------------------------------
    # 2. dormant → archived
    # --------------------------------------------------

    @classmethod
    def _consolidate_to_archived(
        cls,
        npc_id: str,
        current_day: int,
        store,
    ) -> int:
        """
        dormant 且 importance < 0.15 且 28天未访问 → archived。
        """
        events = store.query_by_type(
            memory_type="episodic_event",
            npc_id=npc_id,
            top_k=BULK_QUERY_LIMIT,
            include_archived=False,
        )

        archived_count = 0
        for evt in events:
            meta = evt.get("metadata", {})
            status = meta.get("status", "active")

            # 只处理 dormant
            if status != "dormant":
                continue

            importance = float(meta.get("importance", 0.5))
            if importance >= DEEP_ARCHIVE_IMPORTANCE_THRESHOLD:
                continue

            # 检查 28 天未访问
            days_since_access = _days_since(meta, current_day)
            if days_since_access < DEEP_ARCHIVE_DORMANT_DAYS:
                continue

            store.update_memory(
                memory_id=evt["memory_id"],
                memory_type="episodic_event",
                metadata={
                    "status": "archived",
                    "last_access": current_day,
                },
            )
            archived_count += 1

        print(f"[SeasonEnd] {npc_id}: {archived_count} dormant events → archived")
        return archived_count

    # --------------------------------------------------
    # 3. 合并重复 narrative_arc
    # --------------------------------------------------

    @classmethod
    def _merge_duplicate_narrative_arcs(
        cls,
        npc_id: str,
        current_day: int,
        store,
        llm=None,
    ) -> int:
        """找出语义高度相似的 narrative_arc 并合并。"""
        arcs = store.query_by_type(
            memory_type="narrative_arc",
            npc_id=npc_id,
            top_k=BULK_QUERY_LIMIT,
            include_archived=False,
        )

        if len(arcs) < 2:
            return 0

        # 计算所有 arc 的 embedding
        embed_fn = store.embedding_function.embed_query
        arc_embeddings = [embed_fn(arc["content"]) for arc in arcs]

        # 找出相似对并合并
        merged_ids = set()
        merge_count = 0

        for i in range(len(arcs)):
            if arcs[i]["memory_id"] in merged_ids:
                continue
            for j in range(i + 1, len(arcs)):
                if arcs[j]["memory_id"] in merged_ids:
                    continue
                sim = cls._cosine_similarity(arc_embeddings[i], arc_embeddings[j])
                if sim >= ARC_MERGE_SIMILARITY:
                    merged_arc = cls._do_merge_arcs(
                        arcs[i], arcs[j], npc_id, current_day, store, llm,
                    )
                    if merged_arc:
                        # 物理删除旧的两个 arc
                        store.delete_memory(
                            memory_id=arcs[i]["memory_id"],
                            memory_type="narrative_arc",
                            hard=True,
                        )
                        store.delete_memory(
                            memory_id=arcs[j]["memory_id"],
                            memory_type="narrative_arc",
                            hard=True,
                        )
                        merged_ids.add(arcs[i]["memory_id"])
                        merged_ids.add(arcs[j]["memory_id"])
                        merge_count += 1

        print(f"[SeasonEnd] {npc_id}: merged {merge_count} duplicate narrative_arcs")
        return merge_count

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @classmethod
    def _do_merge_arcs(
        cls,
        arc1: dict,
        arc2: dict,
        npc_id: str,
        current_day: int,
        store,
        llm=None,
    ) -> dict | None:
        """合并两个 narrative_arc，返回新的 merged arc dict。"""
        content1 = arc1["content"]
        content2 = arc2["content"]
        meta1 = arc1.get("metadata", {})
        meta2 = arc2.get("metadata", {})

        # 合并 source_event_ids
        ids1 = _safe_load_json_list(meta1.get("source_event_ids_json", "[]"))
        ids2 = _safe_load_json_list(meta2.get("source_event_ids_json", "[]"))
        merged_source_ids = list(dict.fromkeys(ids1 + ids2))[:30]

        # 合并 relationship_delta (取平均)
        delta1 = _safe_load_json_dict(meta1.get("relationship_delta_json", "{}"))
        delta2 = _safe_load_json_dict(meta2.get("relationship_delta_json", "{}"))
        merged_delta = {}
        for dim in ("trust", "warmth", "familiarity", "confusion", "resentment"):
            v1 = float(delta1.get(dim, 0.0))
            v2 = float(delta2.get(dim, 0.0))
            merged_delta[dim] = _clamp((v1 + v2) / 2.0, -1.0, 1.0)

        # 用 LLM 合并叙事内容（有 LLM 就用，没有就简单拼接）
        merged_content = None
        if llm:
            merged_content = cls._llm_merge_content(llm, npc_id, content1, content2)
        if not merged_content:
            merged_content = f"{content1}\n同时，{content2}"

        # 合并 week_range
        wr1 = meta1.get("week_range", "")
        wr2 = meta2.get("week_range", "")
        merged_week_range = f"{wr1}+{wr2}" if wr1 and wr2 else (wr1 or wr2)

        # 写入新 arc
        new_id = f"arc_merged_{uuid.uuid4().hex[:8]}"
        new_meta = {
            "memory_id": new_id,
            "memory_type": "narrative_arc",
            "npc_id": npc_id,
            "time": current_day,
            "last_access": current_day,
            "location": "Mind Palace",
            "importance": max(
                float(meta1.get("importance", 0.85)),
                float(meta2.get("importance", 0.85)),
            ),
            "status": "active",
            "source": "season_merge",
            "week_range": merged_week_range,
            "source_event_ids_json": json.dumps(merged_source_ids, ensure_ascii=False),
            "relationship_delta_json": json.dumps(merged_delta, ensure_ascii=False),
        }

        store.add_memory(
            memory_type="narrative_arc",
            content=merged_content,
            metadata=new_meta,
            memory_id=new_id,
            upsert=True,
        )
        return {"memory_id": new_id, "content": merged_content, "metadata": new_meta}

    @staticmethod
    def _llm_merge_content(llm, npc_id: str, content1: str, content2: str) -> str | None:
        """调用 LLM 合并两段叙事弧。"""
        prompt = f"""你是 {npc_id} 的潜意识。
以下两段是你在不同周反思中写下的内心独白，但它们高度相似/重复：

【周记A】{content1}

【周记B】{content2}

请将它们合并为一段精炼的内心独白：
- 保留两段中的核心感受和转变
- 去除重复和流水账
- 保持第一人称
- 不要超过两段话

直接输出合并后的文本，不要加任何格式标记。"""

        try:
            res = llm.invoke(prompt)
            if isinstance(res.content, list):
                raw = " ".join(
                    str(item['text']) if isinstance(item, dict) and 'text' in item else ''
                    for item in res.content
                )
            elif isinstance(res.content, dict) and 'text' in res.content:
                raw = str(res.content['text'])
            else:
                raw = str(res.content)
            return raw.strip() or None
        except Exception as e:
            print(f"[SeasonEnd] LLM merge failed: {e}")
            return None

    # --------------------------------------------------
    # 4. 重建 preference_belief confidence
    # --------------------------------------------------

    @classmethod
    def _rebuild_belief_confidence(
        cls,
        npc_id: str,
        current_day: int,
        store,
    ) -> int:
        """
        对所有 preference_belief 做季节性 confidence 重建：
        - confidence 向 0.5 均值回归 (regression to mean)
        - contradiction_count > 0 的 belief 额外衰减
        - confidence < 0.35 的 belief 标记为 dormant
        """
        beliefs = store.query_by_type(
            memory_type="preference_belief",
            npc_id=npc_id,
            top_k=BULK_QUERY_LIMIT,
            include_archived=False,
        )

        rebuilt_count = 0
        for belief in beliefs:
            meta = belief.get("metadata", {})
            old_confidence = float(meta.get("confidence", 0.5))
            contradiction_count = int(meta.get("contradiction_count", 0))
            polarity = meta.get("polarity", "uncertain")

            # 均值回归
            new_confidence = old_confidence + (0.5 - old_confidence) * CONFIDENCE_REGRESSION_RATE

            # 有矛盾的 belief 额外衰减
            if contradiction_count > 0:
                new_confidence -= CONTRADICTION_EXTRA_DECAY * contradiction_count

            new_confidence = _clamp(new_confidence, 0.2, 0.95)

            # 更新 status
            status = meta.get("status", "active")
            if new_confidence < 0.35:
                status = "dormant"
            elif status == "dormant" and new_confidence >= 0.35:
                status = "active"

            # 重建 content
            target = meta.get("target", "player")
            topic = meta.get("topic", "")
            content = (
                f"{npc_id} 目前认为 {target} 对 {topic} 的偏好是 {polarity}，"
                f"置信度为 {new_confidence:.2f}。"
            )

            store.add_memory(
                memory_type="preference_belief",
                content=content,
                metadata={
                    **meta,
                    "confidence": new_confidence,
                    "status": status,
                    "last_access": current_day,
                },
                memory_id=belief["memory_id"],
                upsert=True,
            )
            rebuilt_count += 1

        print(f"[SeasonEnd] {npc_id}: rebuilt {rebuilt_count} belief confidences")
        return rebuilt_count

    # --------------------------------------------------
    # 5. 重建 relationship_impression summary
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

        # 获取本季所有 episodic_events
        events = store.get_events_in_time_range(npc_id, season_start, current_day)
        # 只用 active 事件重建
        active_events = [
            e for e in events
            if e.get("metadata", {}).get("status", "active") == "active"
        ]

        if not active_events:
            print(f"[SeasonEnd] {npc_id}: no active events this season, skip impression rebuild")
            return False

        # 从默认值重新计算
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

            # 读取 relationship_signal（优先级高于 topic_tags）
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
                # 用 topic_tags 推断
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

        # Clamp 到 [0, 1]
        trust = _clamp(trust)
        warmth = _clamp(warmth)
        familiarity = _clamp(familiarity)
        confusion = _clamp(confusion)
        resentment = _clamp(resentment)

        # 最多保留 30 条 evidence
        evidence_event_ids = evidence_event_ids[-30:]

        # Upsert
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
