from typing import Dict,List
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
    """
    计算逻辑：基础权重 * 修正系数
    """
    base = TAG_BASE_WEIGHT.get(tag, 0.1)
    modifier = SAL_MODIFIER.get(salience, 1.0)
    
    # 计算并确保不超过 1.0
    final_weight = round(min(base * modifier, 1.0), 3)
    return final_weight

#=====总结为中期记忆=======
def summarize_to_mid_term(state: dict):
    from memory.STMemory import ChatMemory
    from llm import create_llm
    
    npc_id = state.get("npc_id", "Damon")
    current_time = state.get("game_time", "Unknown")
    current_loc = state.get("location", "Unknown")
    
    # 1. 读取短期历史
    raw_history = ChatMemory.load(npc_id, limit=30)
    if not raw_history:
        return None

    dialogue_text = "\n".join([f"{m['role']}: {m['content']}" for m in raw_history])

    # 2. 构建更严谨的 Prompt
    llm = create_llm()
    prompt = f"""
你现在是游戏 NPC 记忆过滤器。请分析对话记录，将其转为【中期记忆】。

【当前时刻】：{current_time}，地点：{current_loc}
【待处理对话】：
{dialogue_text}

### 任务要求：
1. **第一人称总结**：以 NPC 视角总结，不超过 40 字。
   - **核心约束**：严禁涵盖时间和地点。
   - **强制身份绑定**：对话的另一方必须被称为 **“玩家”** ，不能是一位陌生人等代词。

2. **唯一标签选择**：从以下分类中选出【最符合】的一个：
   - `casual`: 闲聊、问候、天气。
   - `fact`: 涉及玩家喜好、背景。
   - `promise`: 涉及约定、计划或未来要做的承诺。
   - `vulnerability`: 涉及情感剖白、内心秘密、脆弱时刻。
   - `milestone`: 涉及关系重大突破或剧烈冲突。
3. **重要性判定**：在该标签层级内判断重要性（low, medium, high）。

请严格输出 JSON 格式：
{{
  "summary": "...",
  "tag": "...",
  "salience": "..."
}}
"""
    try:
        res = llm.invoke(prompt)

        if isinstance(res.content, list):
            # 提取每个字典中的 'text' 字段，并确保它是字符串类型
            res_content = " ".join(
                str(item['text']) if isinstance(item, dict) and 'text' in item else ''
                for item in res.content
            )
        else:
            # 如果不是列表，而是字典，提取 'text' 字段
            if isinstance(res.content, dict) and 'text' in res.content:
                res_content = str(res.content['text'])
            else:
                res_content = str(res.content)

        #clean_json = res.content.replace("```json", "").replace("```", "").strip()        
        clean_json = res_content.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(clean_json)
        
        # 提取分析结果
        content = analysis.get("summary", "")
        # 我们现在强制要求单标签，以保证权重准确
        chosen_tag = analysis.get("tag", "casual") 
        salience = analysis.get("salience", "medium")

        # 3. 计算最终权重
        final_weight = compute_weight(chosen_tag, salience)

        # 4. 构造标准 MidTermMemoryItem
        mid_item = {
            "time": current_time,
            "location": current_loc,
            "content": content,
            "tags": [chosen_tag], # 存入列表以保持兼容
            "weight": final_weight
        }

        # 5. 写入与清理
        _write_mid_term_file(npc_id, mid_item)
        ChatMemory.clear(npc_id)
        
        return mid_item

    except Exception as e:
        print(f"❌ 中期记忆处理失败: {e}")
        return None

def _write_mid_term_file(npc_id, item):
    """辅助函数：处理 JSON 追加写入"""
    file_path = f"memory/mid_term_{npc_id}.json"
    
    # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []
    
    data.append(item)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
#=====升级为长期记忆=======   
class MidTermMemory:
    # 统一管理文件路径模板，方便修改
    FILE_PATH_TEMPLATE = "memory/mid_term_{npc_id}.json"

    @classmethod
    def load(cls, npc_id: str) -> List[Dict]:
        """
        加载中期记忆文件。
        返回：包含完整信息的字典列表 (content, location, time, weight, tags...)
        """
        file_path = cls.FILE_PATH_TEMPLATE.format(npc_id=npc_id)
        
        if not os.path.exists(file_path):
            return []
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 简单校验，确保是列表且内部是字典
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
                return []
        except Exception as e:
            print(f"❌ [MidTermMemory] 读取失败: {e}")
            return []

    @classmethod
    def clear(cls, npc_id: str):
        """清除特定 NPC 的中期记忆文件（通常在每天结算后调用）"""
        file_path = cls.FILE_PATH_TEMPLATE.format(npc_id=npc_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🧹 [MidTermMemory] 已清理 {npc_id} 的中期记忆")
            except Exception as e:
                print(f"❌ [MidTermMemory] 清理失败: {e}")

    @classmethod
    def upgrade(cls, mid_memories: List[Dict], today_int: int) -> List[Dict]:
        long_term_candidates = []
        
        for item in mid_memories:
            # 安全获取字段，防止 KeyErro
            w = item.get("weight", 0.0)
            tags = item.get("tags", [])
            
            should_save = False

            # --- 1. 筛选策略 ---
            if "fact" in tags and w >= 0.3:
                # 事实类（如喜好）门槛低，高概率保留
                if random.random() < 0.8: should_save = True
            
            elif w > 0.85:
                # 极其重要的记忆（里程碑），100% 保留
                should_save = True
            
            elif w >= 0.45:
                # 普通重要记忆，按权重概率保留
                if random.random() <= w: should_save = True
            
            # --- 2. 格式转换 ---
            if should_save:
                ltm_item = cls._create_ltm_item(item, today_int)
                long_term_candidates.append(ltm_item)

        return long_term_candidates

    @staticmethod
    def _create_ltm_item(item: Dict, today_int: int) -> Dict:  
        return {
            "memory_id": str(uuid.uuid4()),
            "npc_id": item.get("npc_id", "Damon"),
            "content": item.get("content", ""),            
            "time": today_int,        
            "last_access": today_int,
            "location": item.get("location", "Unknown"),             
            "memory_type": "event",
            "importance": round(item.get("weight", 0.5), 3)
        } 
    
    def add_gossip_entry(npc_id, content, new_weight, location, time):
        """
        专用接口：直接向某人的中期记忆列表中插入一条“传闻”
        """
        file_path = f"memory/mid_term_{npc_id}.json"
        
        # 1. 读取现有记忆
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []

        # 2. 构造新的八卦条目
        # 注意：这里我们打个 tag 叫 gossip，方便以后区分
        new_entry = {
            "content": content,
            "weight": round(new_weight, 2), # 保留两位小数
            "location": location,
            "time": time,
            "tags": ["gossip"]
        }

        # 3. 追加并保存
        data.append(new_entry)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)