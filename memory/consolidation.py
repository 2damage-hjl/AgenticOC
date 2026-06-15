"""
WEEKLY_REFLECTION 长期记忆周反思。

职责：
1. 生成 narrative_arc（叙事弧，不是流水账）
2. 计算 relationship_delta（五维关系变化）
3. 标记被吸收的 episodic_event 为 consolidated
"""
import json
import uuid
from typing import Dict, List, Tuple


class WeeklyReflector:
    @classmethod
    def run_weekly_reflection(
        cls,
        npc_id: str,
        current_time: int,
        store,
        llm,
    ) -> dict:
        """
        对过去 7 天的 episodic_event 做周反思。

        Returns:
            {
                "npc_id": str,
                "narrative_arc_id": str,
                "relationship_delta": dict,
            }
            如果无事件，返回 {"npc_id": npc_id, "narrative_arc_id": None, "relationship_delta": {}}
        """
        start_time = current_time - 7
        end_time = current_time

        print(f"[{npc_id}] 正在拉取第 {start_time} 天到第 {end_time} 天的长期记忆...")

        # 1. 从 ChromaDB 拉取长期记忆
        weekly_events = store.get_events_in_time_range(npc_id, start_time, end_time)

        if not weekly_events:
            print(f"[{npc_id}] 过去一周平淡如水，无值得反思的事件。")
            return {
                "npc_id": npc_id,
                "narrative_arc_id": None,
                "relationship_delta": {},
            }

        # 2. 准备 Prompt 上下文
        event_texts = []
        source_event_ids = []
        for evt in weekly_events:
            content = evt["content"]
            importance = evt["metadata"].get("importance", 0.5)
            day = evt["metadata"].get("time", "?")
            event_texts.append(f"- [Day {day}] {content} (重要度:{importance})")

            evt_id = evt["metadata"].get("memory_id") or evt.get("id")
            if evt_id:
                source_event_ids.append(evt_id)

        context = "\n".join(event_texts)

        # 3. LLM 反思：生成叙事弧 + 关系变化
        narrative_text, relationship_delta = cls._call_llm(llm, npc_id, context)

        # 4. 写入 narrative_arc
        narrative_arc_id = None
        if narrative_text:
            narrative_arc_id = f"arc_{uuid.uuid4().hex[:8]}"

            store.add_memory(
                memory_type="narrative_arc",
                content=narrative_text,
                metadata={
                    "memory_id": narrative_arc_id,
                    "memory_type": "narrative_arc",
                    "npc_id": npc_id,
                    "time": current_time,
                    "last_access": current_time,
                    "location": "Mind Palace",
                    "importance": 0.85,
                    "status": "active",
                    "source": "reflection",
                    "week_range": f"{start_time}-{end_time}",
                    "source_event_ids_json": json.dumps(source_event_ids, ensure_ascii=False),
                    "relationship_delta_json": json.dumps(relationship_delta, ensure_ascii=False),
                },
                memory_id=narrative_arc_id,
                upsert=True,
            )

            # 5. 标记被吸收的原始事件为 consolidated
            for evt in weekly_events:
                meta = evt.get("metadata", {})
                evt_id = meta.get("memory_id") or evt.get("id")
                if evt_id:
                    try:
                        store.add_memory(
                            memory_type="episodic_event",
                            content=evt["content"],
                            metadata={**meta, "status": "consolidated"},
                            memory_id=evt_id,
                            upsert=True,
                        )
                    except Exception as e:
                        print(f"[Reflection] Failed to mark {evt_id} as consolidated: {e}")

        return {
            "npc_id": npc_id,
            "narrative_arc_id": narrative_arc_id,
            "relationship_delta": relationship_delta,
        }

    @staticmethod
    def _call_llm(llm, npc_id: str, context: str) -> Tuple[str, dict]:
        """
        调用 LLM 生成叙事弧和关系变化。

        Returns:
            (narrative_text, relationship_delta)
            narrative_text: 叙事弧文本
            relationship_delta: 五维关系变化 dict
        """
        prompt = f"""你是 {npc_id} 的潜意识。
这是过去一周（7天）里，你存入长期记忆的核心事件回顾：

{context}

请进行深度反思：

1. 【narrative】：写一段内心独白式周记。
   - 不要写流水账
   - 要概括这周你和玩家互动的核心模式
   - 写出你心态的转变、困惑或成长
   - 用第一人称

2. 【relationship_delta】：综合这周表现，你的关系感受如何变化？
   - trust: 信任变化 (-1.0 到 1.0)
   - warmth: 亲近变化 (-1.0 到 1.0)
   - familiarity: 熟悉变化 (-1.0 到 1.0)
   - confusion: 困惑变化 (-1.0 到 1.0)
   - resentment: 怨气变化 (-1.0 到 1.0)

严格输出 JSON:
{{
    "narrative": "...",
    "relationship_delta": {{
        "trust": 0.0,
        "warmth": 0.0,
        "familiarity": 0.0,
        "confusion": 0.0,
        "resentment": 0.0
    }}
}}"""

        print(f"[LLM] 正在进行周反思 for {npc_id}...")

        try:
            res = llm.invoke(prompt)
            # 提取文本
            if isinstance(res.content, list):
                raw = " ".join(
                    str(item['text']) if isinstance(item, dict) and 'text' in item else ''
                    for item in res.content
                )
            elif isinstance(res.content, dict) and 'text' in res.content:
                raw = str(res.content['text'])
            else:
                raw = str(res.content)

            # 解析 JSON
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            narrative = parsed.get("narrative", "")
            delta = parsed.get("relationship_delta", {})

            # 规范化 delta
            for dim in ("trust", "warmth", "familiarity", "confusion", "resentment"):
                if dim in delta:
                    delta[dim] = max(-1.0, min(1.0, float(delta[dim])))
                else:
                    delta[dim] = 0.0

            return narrative, delta

        except Exception as e:
            print(f"[Reflection] LLM 解析失败: {e}")
            # fallback：返回简单的叙事
            return f"过去一周 {npc_id} 和玩家有过互动，但反思未能完成。", {}
