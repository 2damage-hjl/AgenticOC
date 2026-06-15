from typing import Dict, List
import random
import uuid
import json
import os

#=====weight======
# 1. 标签的基础权重（阶梯式划分）
TAG_BASE_WEIGHT = {
    "casual": 0.1,        # 闲聊：天气、打招呼
    "fact": 0.35,         # 事实：玩家偏好、个人信息
    "promise": 0.55,      # 承诺：约定、任务、计划
    "vulnerability": 0.75, # 情感：秘密、挫折、深刻交谈
    "milestone": 0.9,     # 里程碑：重大冲突、关系质变、大任完成
}

# 2. 重要性修正系数（让同一标签内部有区分度）
SAL_MODIFIER = {
    "low": 0.8,    # 相对不重要
    "medium": 1.0, # 标准重要
    "high": 1.2,   # 非常重要
}

def compute_weight(tag: str, salience: str) -> float:
    """计算逻辑：基础权重 * 修正系数"""
    base = TAG_BASE_WEIGHT.get(tag, 0.1)
    modifier = SAL_MODIFIER.get(salience, 1.0)
    final_weight = round(min(base * modifier, 1.0), 3)
    return final_weight


#=====scene-level 中期记忆=======
def summarize_to_mid_term(state: dict) -> dict | None:
    """
    END_DIALOGUE 调用：将短期对话压缩为 scene-level 中期候选记忆。
    不写长期记忆，只产出 candidate_events + candidate_beliefs。
    """
    from memory.STMemory import ChatMemory
    from llm import create_llm

    npc_id = state.get("npc_id", "Damon")
    current_time = state.get("game_time", "Unknown")
    current_loc = state.get("location", "Unknown")
    time_num = state.get("time_num", 1)

    # 1. 读取短期历史
    raw_history = ChatMemory.load(npc_id, limit=30)
    if not raw_history:
        return None

    dialogue_text = "\n".join([f"{m['role']}: {m['content']}" for m in raw_history])

    # 2. 构建 Prompt —— 要求输出 scene-level 结构
    llm = create_llm()
    prompt = f"""
你是游戏 NPC「{npc_id}」的记忆系统。请分析以下对话记录，提取【中期候选记忆】。

【当前时刻】：{current_time}，地点：{current_loc}
【待处理对话】：
{dialogue_text}

### 输出要求（严格 JSON）：

1. **scene_id**：生成一个简短的场景标识，如 "day21_town_gift"

2. **candidate_events**：提取值得记住的【具体事件/经历】，每条包含：
   - content：以 NPC 第一人称描述（不超过 40 字），对方称为"玩家"
   - tag：从 casual / fact / promise / vulnerability / milestone 中选一个
   - salience：low / medium / high
   - topic_tags：1-3 个简短主题标签，如 ["gift", "flower"] 或 ["preference", "amethyst"]

3. **candidate_beliefs**：提取值得记住的【偏好、信念、判断】，每条包含：
   - content：以 NPC 第一人称描述（不超过 30 字），如"玩家似乎不喜欢紫水晶"
   - belief_key：简短标识，如 "player_preference_amethyst"
   - confidence：0.0 ~ 1.0（NPC 对此判断的置信度）
   - tag：从 preference / conviction / impression 中选一个

注意：
- 事件和信念是不同维度：事件是"发生了什么"，信念是"我从中得出什么判断"
- 如果对话内容只是闲聊，candidate_events 可以只有 1 条，candidate_beliefs 可以为空
- 不要把时间/地点写进 content

严格输出 JSON：
{{
  "scene_id": "...",
  "candidate_events": [
    {{ "content": "...", "tag": "...", "salience": "...", "topic_tags": ["...", "..."] }}
  ],
  "candidate_beliefs": [
    {{ "content": "...", "belief_key": "...", "confidence": 0.0, "tag": "..." }}
  ]
}}
"""
    try:
        res = llm.invoke(prompt)
        res_content = _extract_text(res)
        clean_json = res_content.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(clean_json)

        # 3. 组装 scene-level 记忆
        scene_id = analysis.get("scene_id", f"day{time_num}_{current_loc}")

        # 为 candidate_events 补齐权重
        candidate_events = []
        for evt in analysis.get("candidate_events", []):
            tag = evt.get("tag", "casual")
            salience = evt.get("salience", "medium")
            evt["weight"] = compute_weight(tag, salience)
            candidate_events.append(evt)

        # 为 candidate_beliefs 补齐
        candidate_beliefs = []
        for blf in analysis.get("candidate_beliefs", []):
            blf["weight"] = round(min(blf.get("confidence", 0.5) * 0.6, 1.0), 3)
            candidate_beliefs.append(blf)

        scene = {
            "scene_id": scene_id,
            "npc_id": npc_id,
            "day": time_num,
            "location": current_loc,
            "candidate_events": candidate_events,
            "candidate_beliefs": candidate_beliefs,
        }

        # 4. 写入中期记忆 + 清空短期
        _write_mid_term_file(npc_id, scene)
        ChatMemory.clear(npc_id)

        return scene

    except Exception as e:
        print(f"❌ 中期记忆处理失败: {e}")
        return None


def _extract_text(res) -> str:
    """从 LLM 响应中提取纯文本。"""
    if isinstance(res.content, list):
        return " ".join(
            str(item['text']) if isinstance(item, dict) and 'text' in item else ''
            for item in res.content
        )
    if isinstance(res.content, dict) and 'text' in res.content:
        return str(res.content['text'])
    return str(res.content)


def _write_mid_term_file(npc_id: str, scene: dict):
    """辅助函数：将 scene 追加写入中期记忆 JSON 文件。"""
    file_path = f"memory/mid_term_{npc_id}.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []

    data.append(scene)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


#=====升级为长期记忆=======
class MidTermMemory:
    FILE_PATH_TEMPLATE = "memory/mid_term_{npc_id}.json"

    @classmethod
    def load(cls, npc_id: str) -> List[Dict]:
        """加载中期记忆文件（scene-level 格式）。"""
        file_path = cls.FILE_PATH_TEMPLATE.format(npc_id=npc_id)
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
                return []
        except Exception as e:
            print(f"❌ [MidTermMemory] 读取失败: {e}")
            return []

    @classmethod
    def clear(cls, npc_id: str):
        """清除特定 NPC 的中期记忆文件。"""
        file_path = cls.FILE_PATH_TEMPLATE.format(npc_id=npc_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🧹 [MidTermMemory] 已清理 {npc_id} 的中期记忆")
            except Exception as e:
                print(f"❌ [MidTermMemory] 清理失败: {e}")

    @classmethod
    def upgrade(cls, mid_memories: List[Dict], today_int: int) -> List[Dict]:
        """
        从 scene-level 中期记忆中提取 candidate_events + candidate_beliefs，
        筛选后转为长期记忆候选列表。
        """
        long_term_candidates = []

        for scene in mid_memories:
            npc_id = scene.get("npc_id", "Damon")
            location = scene.get("location", "Unknown")
            day = scene.get("day", today_int)

            # --- 处理 candidate_events → episodic_event ---
            for evt in scene.get("candidate_events", []):
                w = evt.get("weight", 0.0)
                tag = evt.get("tag", "casual")

                should_save = _should_save_event(w, tag)
                if should_save:
                    topic_tags = evt.get("topic_tags", [])
                    candidates = cls._create_ltm_item(
                        item=evt, today_int=day, npc_id=npc_id, location=location,
                        memory_type="episodic_event", source="dialogue",
                        extra_fields={"topic_tags": topic_tags},
                    )
                    long_term_candidates.append(candidates)

            # --- 处理 candidate_beliefs → preference_belief ---
            for blf in scene.get("candidate_beliefs", []):
                w = blf.get("weight", 0.0)
                confidence = blf.get("confidence", 0.5)

                # 偏好/信念：只要 confidence >= 0.4 且非 trivial，就保留
                if confidence >= 0.4 and w >= 0.2:
                    candidates = cls._create_ltm_item(
                        item=blf, today_int=day, npc_id=npc_id, location=location,
                        memory_type="preference_belief", source="dialogue",
                        extra_fields={"belief_key": blf.get("belief_key", ""), "confidence": confidence},
                    )
                    long_term_candidates.append(candidates)

        return long_term_candidates

    @staticmethod
    def _create_ltm_item(
        item: Dict,
        today_int: int,
        npc_id: str = "Damon",
        location: str = "Unknown",
        memory_type: str = "episodic_event",
        source: str = "dialogue",
        extra_fields: Dict | None = None,
    ) -> Dict:
        base = {
            "memory_id": str(uuid.uuid4()),
            "npc_id": npc_id,
            "content": item.get("content", ""),
            "time": today_int,
            "last_access": today_int,
            "location": location,
            "memory_type": memory_type,
            "importance": round(item.get("weight", 0.5), 3),
            "status": "active",
            "source": source,
        }
        if extra_fields:
            base.update(extra_fields)
        return base

    @staticmethod
    def add_gossip_entry(npc_id, content, new_weight, location, time):
        """专用接口：直接向某人的中期记忆列表中插入一条"传闻"。"""
        file_path = f"memory/mid_term_{npc_id}.json"

        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []

        # 传闻作为独立 scene 写入
        gossip_scene = {
            "scene_id": f"gossip_day{time}",
            "npc_id": npc_id,
            "day": time,
            "location": location,
            "candidate_events": [
                {"content": content, "tag": "gossip", "salience": "medium", "weight": round(new_weight, 2)}
            ],
            "candidate_beliefs": [],
        }
        data.append(gossip_scene)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _should_save_event(weight: float, tag: str) -> bool:
    """筛选策略：决定是否将中期事件升级为长期。"""
    if tag == "fact" and weight >= 0.3:
        if random.random() < 0.8:
            return True
    elif weight > 0.85:
        return True  # milestone，100% 保留
    elif weight >= 0.45:
        if random.random() <= weight:
            return True
    return False