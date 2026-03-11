class WeeklyReflector:
    @classmethod
    def run_weekly_reflection(cls, npc_id: str, current_time: int, store, llm) -> int:
        """
        :param current_time: 当前游戏天数 (例如 7, 14, 21)
        :return: affinity_delta
        """
        # 1. 计算时间窗口 (过去7天)
        # 假设 time 单位是“天数”(int)，如果是时间戳请自行调整
        start_time = current_time - 7 
        end_time = current_time
        
        print(f"[{npc_id}] 正在拉取第 {start_time} 天到第 {end_time} 天的长期记忆...")

        # 2. 从 ChromaDB 拉取长期记忆
        weekly_events = store.get_events_in_time_range(npc_id, start_time, end_time)

        if not weekly_events:
            print(f"[{npc_id}] 过去一周平淡如水，无值得反思的事件。")
            return 0

        # 3. 准备 Prompt 上下文
        # 提取 content 和 importance
        event_texts = []
        for evt in weekly_events:
            content = evt["content"]
            importance = evt["metadata"].get("importance", 0.5)
            day = evt["metadata"].get("time", "?")
            event_texts.append(f"- [Day {day}] {content} (重要度:{importance})")
        
        context = "\n".join(event_texts)

        # 4. LLM 反思
        summary_text, affinity_delta = cls._call_llm(llm, npc_id, context)

        # 5. 存入“周总结”到向量库
        if summary_text:
            import uuid
            store.add(
                layer="summary", 
                content=summary_text,
                metadata={
                    "npc_id": npc_id,
                    "time": current_time,
                    "location": "Mind Palace",
                    "importance": 6.0,          # 周总结权重很高
                    "memory_type": "weekly_summary",
                    "week_range": f"{start_time}-{end_time}",
                    "affinity_change": affinity_delta
                },
                doc_id=f"week_sum_{uuid.uuid4().hex[:8]}"
            )
            
        return affinity_delta

    @staticmethod
    def _call_llm(llm, npc_id, context):
        prompt = f"""
你是 {npc_id} 的潜意识。
这是过去一周（7天）里，你存入长期记忆的核心事件回顾：

{context}

请进行深度反思：
1. 【Summary】：请写一段周记。不要写流水账，要概括这周你和玩家互动的核心模式，以及你心态的转变。
2. 【Affinity】：综合这周的所有表现，你对玩家的好感度要如何调整？(-400 到 +400)

严格输出 JSON:
{{
    "summary": "...",
    "affinity_delta": 0
}}
"""
        # 模拟 LLM 返回 (请替换为真实调用)
        print(f"🤖 [LLM] 正在进行周反思...")
        return "这周玩家送了我很多好东西，我觉得他是个值得信赖的人。", 30