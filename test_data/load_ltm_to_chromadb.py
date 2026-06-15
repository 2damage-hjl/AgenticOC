"""
将 ltm_data.json 导入临时 ChromaDB，返回 MemoryStore 实例。

用法:
    from test_data.load_ltm_to_chromadb import load_test_ltm
    store, tmpdir = load_test_ltm("test_data/ltm_data.json")
    # ... 跑评估 ...
    import shutil; shutil.rmtree(tmpdir)
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Tuple

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_test_ltm(ltm_path: str | None = None) -> Tuple:
    """
    将测试 LTM 数据导入临时 ChromaDB。

    Args:
        ltm_path: ltm_data.json 路径，默认 test_data/ltm_data.json

    Returns:
        (MemoryStore, tempdir_path)
    """
    # 延迟导入，避免启动时加载重型依赖
    from memory.embedded import MemoryStore

    if ltm_path is None:
        ltm_path = Path(__file__).parent / "ltm_data.json"
    else:
        ltm_path = Path(ltm_path)

    with open(ltm_path, "r", encoding="utf-8") as f:
        all_ltm = json.load(f)

    # 创建临时 ChromaDB
    tmpdir = tempfile.mkdtemp(prefix="eval_chroma_")
    store = MemoryStore(db_path=tmpdir)

    total = 0
    for npc_id, npc_data in all_ltm.items():
        for memory_type, memories in npc_data.items():
            for mem in memories:
                # 分离 metadata 和 content
                metadata = {
                    "npc_id": mem["npc_id"],
                    "time": mem["time"],
                    "location": mem["location"],
                    "importance": mem["importance"],
                    "status": mem.get("status", "active"),
                    "source": mem.get("source", "test_data"),
                }
                # 复制额外字段（topic_tags, emotional_valence 等）
                for key in mem:
                    if key not in ("memory_id", "memory_type", "npc_id", "content",
                                   "time", "location", "importance", "status",
                                   "last_access", "source"):
                        val = mem[key]
                        if isinstance(val, list):
                            # ChromaDB metadata 只接受 str/int/float/bool
                            metadata[f"{key}_json"] = json.dumps(val, ensure_ascii=False)
                        elif isinstance(val, dict):
                            metadata[f"{key}_json"] = json.dumps(val, ensure_ascii=False)
                        else:
                            metadata[key] = val

                store.add_memory(
                    memory_type=memory_type,
                    content=mem["content"],
                    metadata=metadata,
                    memory_id=mem["memory_id"],
                )
                total += 1

    print(f"[load_test_ltm] 已导入 {total} 条记忆到 {tmpdir}")
    return store, tmpdir


if __name__ == "__main__":
    store, tmpdir = load_test_ltm()
    print(f"临时 ChromaDB 路径: {tmpdir}")
    print("验证：各 collection 行数")
    for mt in store.MEMORY_TYPES:
        count = store._get_collection(mt)._collection.count()
        print(f"  {mt}: {count}")
