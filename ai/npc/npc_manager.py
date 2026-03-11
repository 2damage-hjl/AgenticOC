import json
import os
from typing import Tuple

def load_relationship_config(npc_id: str, relationship_status: str) -> Tuple[str, str, str]:
    """
    读取 NPC 配置文件，返回特定关系下的 (description, instruction, example)
    
    Args:
        npc_id: NPC 名字，如 "Damon"
        relationship_status: 关系状态，如 "friend"

    Returns:
        tuple: (description, instruction, example)
    """
    # 1. 构造路径
    file_path = f"npc/{npc_id}.json"
    
    # 默认值（防止文件缺失或状态匹配失败）
    default_data = {
        "description": "普通关系。",
        "instruction": "正常交流。",
        "example": "你好。"
    }

    # 2. 读取 JSON 文件
    if not os.path.exists(file_path):
        print(f"⚠️ [Config] 未找到 {npc_id} 的配置文件，使用默认值。")
        data_map = {}
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                data_map = config.get("relationship_map", {})
        except Exception as e:
            print(f"❌ [Config] 读取 {npc_id} 配置出错: {e}")
            data_map = {}

    # 3. 获取特定关系的数据 (兼容大小写)
    key = relationship_status.lower() # 转小写匹配
    
    # 如果找不到该关系（比如传入了 'enemy' 但json里没有），默认回退到 'stranger'
    if key not in data_map:
        # print(f"⚠️ [Config] {npc_id} 没有定义 '{key}' 关系，回退到 stranger。")
        target_data = data_map.get("stranger", default_data)
    else:
        target_data = data_map[key]

    # 4. 分开返回
    return (
        target_data.get("description", ""),
        target_data.get("instruction", ""),
        target_data.get("example", "")
    )