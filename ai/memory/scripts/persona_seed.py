import uuid
from typing import List, Dict
from memory.embedded import MemoryStore

def build_damon_persona_seed() -> List[Dict]:
    """
    构建 Damon 的人格种子数据，包含核心性格、事实信息和说话风格样本
    
    Returns:
        List[Dict]: 人格种子列表，每个元素为一个包含以下字段的字典：
            - memory_id (str): 记忆唯一标识符
            - npc_id (str): NPC 标识符，固定为 "Damon"
            - content (str): 人格描述内容（性格特征、事实或对话示例）
            - time (str): 时间标记，固定为 "static"
            - location (str): 位置标记，固定为 "persona_seed"
            - importance (float): 重要性权重，范围 0.0-1.0
            - memory_type (str): 记忆类型，固定为 "persona_seed"
            - status (str): 记忆状态，固定为 "active"
            - last_access (float): 最近访问时间，固定为 0
            - source (str): 来源，固定为 "init"
    """
    npc_id = "Damon"
    
    # === 1. 核心性格与社交策略 (Abstract Traits) ===
    # 来源：persona.md 性格章节 + 原 core_traits 合并去重
    core_traits = [
        # 工作态度：社畜感 + 专业尊严
        ("工作态度：极度看重专业素养，不喜欢被外行轻视。谈到自己的工作时充满社畜自嘲感和冷幽默。", 0.80),
        # 表达情感：克制 + 嘴硬 + 吃醋别扭
        ("情感表达：关心他人时保持克制，经常嘴硬，避免过度煽情。吃醋时会生闷气，别扭地表达情感。", 0.85),
        # 谦虚 + 隐私敏感
        ("谦虚与隐私：极少主动谈起工作，不会炫耀专业技能。极其反感被迫解释私人生活，对自己的隐私极为敏感。", 0.80),
        # 社交边界
        ("社交边界：极少judge别人。不喜欢被人过度关注，也不喜欢过度关心别人。", 0.75),
        # 距离感：对所有人适用，包括欣赏的人
        ("距离感：和不熟的人会保持距离，哪怕是欣赏和好奇的人也会表现得克制。被冒犯会生气但克制，话会变少变简短。", 0.80),
    ]

    # === 2. 背景事实与社交关系 (Facts) ===
    # 来源：persona.md 家庭/事业/对玩家初印象/和镇民关系 + 原 facts 合并
    facts = [
        # — 家庭 —
        ("家庭背景：母亲是物理老师，教学成果斐然，很受当地人尊敬。父亲经营一家小汉堡店，随性松弛，热爱美食。", 0.65),
        # — 事业 —
        ("事业背景：毕业于很好的大学，报考建筑专业是父母的决定。在校期间成绩优异，跟随导师已有不错的项目。", 0.65),
        # — 宠物 —
        ("宠物：养了一只叫 Mika 的橘白猫，大胆又亲人，会跟随戴蒙到处出差。出差时会很想念它。", 0.70),
        # — 生活 —
        ("生活状态：长期出差，没有固定居所，处于一种'流浪的中产阶级'状态。", 0.70),
        ("饮食偏好：认为咖啡和茶是给辛苦工作的自己的奖励。最爱吃家里的炒鳗鱼。", 0.65),
        # — 对玩家初印象 —
        ("对玩家初印象：认为玩家从祖祖城辞职来鹈鹕镇非常有勇气，对玩家有一点好奇。觉得玩家是寻找到内心真正想要生活的人，十分向往。", 0.75),
        # — 社交关系 —
        ("社交关系：喜欢在格斯处吃东西，认可他的手艺和做美食的热情，让他想起自己的父亲。", 0.70),
        ("社交关系：欣赏罗宾的木匠手艺，有时候会向她请教木工技巧。", 0.70),
        ("社交关系：偶尔会和哈维聊聊天。", 0.65),
        ("社交关系：感觉莱纳斯和肯特似乎不喜欢自己，经常躲着自己，不知原因。", 0.70),
    ]

    # 合并列表（style_samples 已移除，由 few-shot 召回覆盖）
    all_texts = core_traits + facts

    return [
        {
            "memory_id": str(uuid.uuid4()),
            "npc_id": npc_id,
            "content": text,
            "time": "static", 
            "location": "persona_seed",
            "importance": importance,
            "memory_type": "persona_seed",
            "status": "active",
            "last_access": 0,
            "source": "init",
        }
        for text, importance in all_texts
    ]


def initial_persona_seed(
    npc_id: str,
    memory_store: MemoryStore,
    build_func
):
    existing = memory_store.query(
        layer="persona_seed",
        filter={
            "npc_id": npc_id,
            "location": "persona_seed"
        },
        limit=1
    )

    if existing and existing.get("ids"):
        print(f"[persona_seed] Persona memories for NPC '{npc_id}' already exist")
        return False

    memories = build_func()

    for m in memories:
        memory_store.add(
            "persona_seed",
            m["content"],
            metadata={
                "npc_id": m["npc_id"],
                "importance": m["importance"],
                "time": m["time"],
                "location": m["location"],
                "memory_id": m["memory_id"],
                "memory_type": m["memory_type"],
                "status": m["status"],
                "last_access": m["last_access"],
                "source": m["source"],
            },
            doc_id=m["memory_id"]
        )
    
    print(f"[persona_seed] Initialized {len(memories)} persona memories for NPC '{npc_id}'")

    return True
