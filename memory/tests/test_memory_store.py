"""
MemoryStore 长期记忆管理接口验收测试。

运行方式：
    cd d:/DamonAI/ai
    python -m memory.tests.test_memory_store
"""
import os
import sys
import tempfile
import shutil

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _fresh_store():
    """创建一个使用临时目录的干净 MemoryStore，避免污染正式数据。"""
    from memory.embedded import MemoryStore
    tmp = tempfile.mkdtemp(prefix="test_chroma_")
    store = MemoryStore(db_path=tmp)
    return store, tmp


def _cleanup(store, tmp):
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass


def test_add_memory():
    """测试 1：新增记忆"""
    store, tmp = _fresh_store()
    try:
        memory_id = store.add_memory(
            memory_type="episodic_event",
            content="玩家告诉 Damon 自己讨厌紫水晶。",
            metadata={
                "npc_id": "Damon",
                "time": 5,
                "location": "Town",
                "importance": 0.7,
            },
        )
        mem = store.get_memory(memory_id, memory_type="episodic_event")
        assert mem is not None, "get_memory 返回 None"
        assert mem["metadata"]["status"] == "active", f"status 应为 active，实际: {mem['metadata']['status']}"
        assert mem["metadata"]["memory_type"] == "episodic_event"
        assert mem["metadata"]["memory_id"] == memory_id
        print("✅ 测试 1 通过：add_memory + get_memory")
    finally:
        _cleanup(store, tmp)


def test_update_memory():
    """测试 2：更新记忆"""
    store, tmp = _fresh_store()
    try:
        memory_id = store.add_memory(
            memory_type="episodic_event",
            content="玩家告诉 Damon 自己讨厌紫水晶。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.7},
        )
        ok = store.update_memory(
            memory_id,
            metadata={"importance": 0.4, "status": "consolidated"},
            memory_type="episodic_event",
        )
        assert ok, "update_memory 返回 False"

        mem = store.get_memory(memory_id, memory_type="episodic_event")
        assert mem["metadata"]["importance"] == 0.4
        assert mem["metadata"]["status"] == "consolidated"
        # 确保旧字段不丢失
        assert mem["metadata"]["npc_id"] == "Damon"
        assert mem["metadata"]["time"] == 5
        assert mem["metadata"]["location"] == "Town"
        print("✅ 测试 2 通过：update_memory 保留旧字段")
    finally:
        _cleanup(store, tmp)


def test_query_by_type():
    """测试 3：按类型查询"""
    store, tmp = _fresh_store()
    try:
        store.add_memory(
            memory_type="episodic_event",
            content="玩家告诉 Damon 自己讨厌紫水晶。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.7},
        )
        store.update_memory(
            store.collections["episodic_event"]._collection.get(
                include=["metadatas"], limit=1
            )["ids"][0] if False else None,
            # 用 query 找到 id
            memory_type="episodic_event",
        ) if False else None  # placeholder

        # 先拿到刚才添加的 id
        results = store.query_by_type(
            memory_type="episodic_event",
            npc_id="Damon",
        )
        assert len(results) >= 1, "应该至少查到 1 条"

        # 改状态后再查
        memory_id = results[0]["memory_id"]
        store.update_memory(memory_id, metadata={"status": "consolidated"}, memory_type="episodic_event")

        filtered = store.query_by_type(
            memory_type="episodic_event",
            npc_id="Damon",
            where={"status": "consolidated"},
        )
        assert len(filtered) >= 1, "应该查到 consolidated 记忆"
        print("✅ 测试 3 通过：query_by_type 按条件过滤")
    finally:
        _cleanup(store, tmp)


def test_archive_memory():
    """测试 4：软删除"""
    store, tmp = _fresh_store()
    try:
        memory_id = store.add_memory(
            memory_type="episodic_event",
            content="测试归档。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.3},
        )
        ok = store.archive_memory(memory_id, memory_type="episodic_event", reason="low_importance")
        assert ok, "archive_memory 返回 False"

        mem = store.get_memory(memory_id, memory_type="episodic_event")
        assert mem["metadata"]["status"] == "archived"
        assert mem["metadata"]["archived_reason"] == "low_importance"
        print("✅ 测试 4 通过：archive_memory 设置 status 和 reason")
    finally:
        _cleanup(store, tmp)


def test_query_excludes_archived():
    """测试 5：默认查询不返回 archived"""
    store, tmp = _fresh_store()
    try:
        store.add_memory(
            memory_type="episodic_event",
            content="会被归档的。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.2},
        )
        # 找到 id 并归档
        results = store.query_by_type(memory_type="episodic_event", npc_id="Damon", include_archived=True)
        memory_id = results[0]["memory_id"]
        store.archive_memory(memory_id, memory_type="episodic_event", reason="test")

        # 默认不返回 archived
        normal = store.query_by_type(memory_type="episodic_event", npc_id="Damon")
        assert all(r["metadata"].get("status") != "archived" for r in normal), "默认查询不应包含 archived"
        print("✅ 测试 5 通过：默认查询排除 archived")
    finally:
        _cleanup(store, tmp)


def test_query_include_archived():
    """测试 6：include_archived=True 可查到"""
    store, tmp = _fresh_store()
    try:
        store.add_memory(
            memory_type="episodic_event",
            content="归档后查询测试。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.2},
        )
        results = store.query_by_type(memory_type="episodic_event", npc_id="Damon", include_archived=True)
        memory_id = results[0]["memory_id"]
        store.archive_memory(memory_id, memory_type="episodic_event", reason="test")

        with_archived = store.query_by_type(memory_type="episodic_event", npc_id="Damon", include_archived=True)
        archived_ids = [r["memory_id"] for r in with_archived if r["metadata"].get("status") == "archived"]
        assert memory_id in archived_ids, "include_archived=True 应能查到"
        print("✅ 测试 6 通过：include_archived=True 可查到")
    finally:
        _cleanup(store, tmp)


def test_hard_delete():
    """测试 7：hard delete"""
    store, tmp = _fresh_store()
    try:
        memory_id = store.add_memory(
            memory_type="episodic_event",
            content="将被物理删除。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.1},
        )
        ok = store.delete_memory(memory_id, memory_type="episodic_event", hard=True)
        assert ok, "delete_memory hard=True 返回 False"

        mem = store.get_memory(memory_id, memory_type="episodic_event")
        assert mem is None, "物理删除后 get_memory 应返回 None"
        print("✅ 测试 7 通过：hard delete 后 get_memory 返回 None")
    finally:
        _cleanup(store, tmp)


def test_soft_delete():
    """测试 8：soft delete 等同 archive"""
    store, tmp = _fresh_store()
    try:
        memory_id = store.add_memory(
            memory_type="episodic_event",
            content="软删除测试。",
            metadata={"npc_id": "Damon", "time": 5, "location": "Town", "importance": 0.1},
        )
        ok = store.delete_memory(memory_id, memory_type="episodic_event", hard=False)
        assert ok

        mem = store.get_memory(memory_id, memory_type="episodic_event")
        assert mem["metadata"]["status"] == "archived"
        assert mem["metadata"]["archived_reason"] == "soft_delete"
        print("✅ 测试 8 通过：soft delete 等同 archive")
    finally:
        _cleanup(store, tmp)


def test_add_compatibility():
    """测试 9：旧 add() 接口兼容"""
    store, tmp = _fresh_store()
    try:
        mid = store.add(
            layer="episodic_event",
            content="旧接口写入。",
            metadata={"npc_id": "Damon", "time": 1, "location": "Farm", "importance": 0.5},
            doc_id="old_iface_001",
        )
        assert mid == "old_iface_001"
        mem = store.get_memory("old_iface_001", memory_type="episodic_event")
        assert mem is not None
        assert mem["metadata"]["status"] == "active"
        print("✅ 测试 9 通过：旧 add() 接口兼容")
    finally:
        _cleanup(store, tmp)


if __name__ == "__main__":
    test_add_memory()
    test_update_memory()
    test_query_by_type()
    test_archive_memory()
    test_query_excludes_archived()
    test_query_include_archived()
    test_hard_delete()
    test_soft_delete()
    test_add_compatibility()
    print("\n🎉 全部验收测试通过！")
