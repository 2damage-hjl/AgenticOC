import json
import os

#======短期记忆=======
class ChatMemory:
    @staticmethod
    def get_path(npc_id: str):
        return f"chat_history_{npc_id}.json"

    @staticmethod
    def load(npc_id: str, limit: int = 10):
        """加载最近的聊天记录"""
        file_path = ChatMemory.get_path(npc_id)
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                return history[-limit:]
        except Exception:
            return []

    @staticmethod
    def save(npc_id: str, role: str, content: str):
        """保存单条消息"""
        file_path = ChatMemory.get_path(npc_id)
        history = ChatMemory.load(npc_id, limit=999) # 读取全部以供追加
        history.append({"role": role, "content": content})
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    @staticmethod
    def clear(npc_id: str):
        """清除特定 NPC 的聊天记录"""
        file_path = ChatMemory.get_path(npc_id)
        if os.path.exists(file_path):
            os.remove(file_path)
