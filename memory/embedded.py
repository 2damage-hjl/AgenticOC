import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil
import uuid
import time as _time
from typing import List, Dict, Optional
from langchain_core.documents import Document


class MemoryStore:
    # 已知的长期记忆 collection 名称
    MEMORY_TYPES = [
        "persona_seed",
        "episodic_event",
        "preference_belief",
        "relationship_impression",
        "narrative_arc",
    ]

    # memory_type → 自动 ID 前缀
    _ID_PREFIX = {
        "persona_seed": "persona",
        "episodic_event": "evt",
        "preference_belief": "belief",
        "relationship_impression": "rel",
        "narrative_arc": "arc",
    }

    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self._embeddings = None
        self._collections = {}  # 延迟创建：等 embedding 模型加载后再初始化

    # =========================
    #  内部工具
    # =========================

    @property
    def collections(self):
        """延迟初始化：首次访问时加载 embedding 模型并创建 ChromaDB collection。
        必须在 embedding 模型加载后创建 collection，否则 ChromaDB 会用默认
        all-MiniLM (384维) 锁定 collection 维度，导致 BGE-M3 (1024维) 不兼容。"""
        if not self._collections:
            # 确保 embedding 模型已加载
            _ = self.embedding_function
            # 用 BGE-M3 创建所有 collection（维度正确匹配 1024）
            for mt in self.MEMORY_TYPES:
                self._collections[mt] = self._create_collection(self.db_path, mt)
        return self._collections

    @property
    def embedding_function(self):
        """懒加载 BGE-M3 模型"""
        if self._embeddings is None:
            print("--- [Lazy Load] 正在初始化重型模型 BGE-M3 ---")
            from sentence_transformers import SentenceTransformer
            from langchain_core.embeddings import Embeddings

            class InnerEmbedder(Embeddings):
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)
                def embed_documents(self, texts: List[str]):
                    return self.model.encode(texts, normalize_embeddings=True).tolist()
                def embed_query(self, text: str):
                    return self.model.encode(text, normalize_embeddings=True).tolist()

            self._embeddings = InnerEmbedder("BAAI/bge-m3")

        return self._embeddings

    def _create_collection(self, db_path, collection_name):
        """创建 ChromaDB collection（调用前必须已加载 embedding 模型）"""
        from langchain_chroma import Chroma
        coll = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=self._embeddings,
        )
        # 检测旧 ChromaDB 的向量维度不兼容（旧模型 all-MiniLM = 384 维）
        try:
            existing = coll._collection.get(limit=1, include=["embeddings"])
            if existing and existing.get("embeddings") and existing["embeddings"][0]:
                old_dim = len(existing["embeddings"][0])
                if old_dim != 1024:
                    print(f"\n{'='*60}")
                    print(f"[FATAL] ChromaDB 向量维度不兼容！")
                    print(f"  当前模型 BGE-M3: 1024 维")
                    print(f"  现有数据 {collection_name}: {old_dim} 维（旧模型）")
                    print(f"  请执行以下步骤重建：")
                    print(f"  1. 删除 chroma_db/ 目录")
                    print(f"  2. 删除 .persona_init_done 文件")
                    print(f"  3. 重新启动 server.py")
                    print(f"{'='*60}\n")
                    raise RuntimeError(
                        f"ChromaDB dimension mismatch: BGE-M3(1024) vs existing({old_dim}). "
                        f"Delete chroma_db/ and .persona_init_done, then restart."
                    )
        except RuntimeError:
            raise
        except Exception:
            pass  # 空 collection 或首次创建
        return coll

    def _get_collection(self, memory_type: str):
        """根据 memory_type 返回对应的 LangChain Chroma collection。"""
        if memory_type not in self.collections:
            raise KeyError(f"未知 memory_type: '{memory_type}'，可用: {list(self.collections.keys())}")
        return self.collections[memory_type]

    def _iter_memory_collections(self):
        """遍历所有已知长期记忆 collection，yield (memory_type, collection)。"""
        for mt, col in self.collections.items():
            yield mt, col

    def _chroma_get_by_id(self, collection, memory_id: str):
        """从单个 collection 按 id 获取一条记录，返回原生 Chroma 结果或 None。"""
        try:
            result = collection._collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if result["ids"]:
                return result
        except Exception:
            pass
        return None

    def _ensure_embedding(self):
        """确保 embedding_function 已加载。"""
        _ = self.embedding_function

    @classmethod
    def _auto_id(cls, memory_type: str) -> str:
        """生成带类型前缀的自动 ID，如 evt_a1b2c3d4。"""
        prefix = cls._ID_PREFIX.get(memory_type, "mem")
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    # =========================
    #  兼容旧接口
    # =========================

    def clear_database(self):
        """物理删除数据库文件夹，彻底重置"""
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"已清理数据库目录: {self.db_path}")
            if os.path.exists(".persona_init_done"):
                os.remove(".persona_init_done")

    def add(self, layer: str, content: str, metadata: Optional[Dict]=None, doc_id: Optional[str]=None):
        """旧接口兼容：委托给 add_memory。"""
        return self.add_memory(
            memory_type=layer,
            content=content,
            metadata=metadata or {},
            memory_id=doc_id,
        )

    def upsert(self, layer: str, content: str, metadata: Optional[Dict]=None, doc_id: Optional[str]=None):
        """旧接口兼容：委托给 add_memory(upsert=True)。"""
        return self.add_memory(
            memory_type=layer,
            content=content,
            metadata=metadata or {},
            memory_id=doc_id,
            upsert=True,
        )

    def delete(self, layer: str, ids: List[str]):
        """旧接口兼容：物理删除。"""
        self.collections[layer]._collection.delete(ids=ids)

    def similarity_search(self, layer: str, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        return self.collections[layer].similarity_search(query=query, k=k, filter=filter)

    def query(self, layer, filter=None, limit=10):
        formatted_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append({key: {"$eq": value}})
            if len(conditions) > 1:
                formatted_filter = {"$and": conditions}
            else:
                formatted_filter = conditions[0]
        return self.collections[layer].get(where=formatted_filter, limit=limit)

    def raw_query(self, layer: str, query_emb: List[float], top_k: int, filter: Optional[Dict] = None):
        """直接调用 Chroma 底层的 query 方法，绕过 LangChain 的封装。"""
        chroma_collection = self.collections[layer]._collection
        formatted_filter = None
        if filter:
            conditions = [{k: {"$eq": v}} for k, v in filter.items()]
            if len(conditions) > 1:
                formatted_filter = {"$and": conditions}
            elif conditions:
                formatted_filter = conditions[0]
        return chroma_collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=formatted_filter
        )

    def get_events_in_time_range(self, npc_id: str, start_time: int, end_time: int) -> List[Dict]:
        """获取指定时间范围内的 EpisodicEvent (用于周结)"""
        where_filter = {
            "$and": [
                {"npc_id": {"$eq": npc_id}},
                {"time": {"$gte": start_time}},
                {"time": {"$lte": end_time}},
                {"memory_type": {"$eq": "episodic_event"}}
            ]
        }
        results = self.collections["episodic_event"]._collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )
        events = []
        if results["documents"]:
            for i in range(len(results["documents"])):
                events.append({
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
        return events

    def get_active_npc_ids(self, start_time: int, end_time: int) -> List[str]:
        """获取指定时间段内有过 EpisodicEvent 记录的所有 NPC ID"""
        results = self.collections["episodic_event"]._collection.get(
            where={
                "$and": [
                    {"time": {"$gte": start_time}},
                    {"time": {"$lte": end_time}},
                    {"memory_type": {"$eq": "episodic_event"}}
                ]
            },
            include=["metadatas"]
        )
        active_ids = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if "npc_id" in meta:
                    active_ids.add(meta["npc_id"])
        return list(active_ids)

    # =========================
    #  新增：长期记忆管理接口
    # =========================

    # ---------- 1. add_memory ----------

    def add_memory(
        self,
        memory_type: str,
        content: str,
        metadata: dict,
        memory_id: str | None = None,
        upsert: bool = False,
    ) -> str:
        """
        新增一条长期记忆，自动补齐 metadata 并写入对应 collection。

        Returns:
            memory_id
        """
        self._ensure_embedding()

        # 生成 ID
        if not memory_id:
            memory_id = self._auto_id(memory_type)

        # 补齐 metadata
        metadata = dict(metadata)  # 复制，不修改调用方
        metadata["memory_id"] = memory_id
        metadata["memory_type"] = memory_type
        metadata.setdefault("status", "active")
        metadata.setdefault("last_access", metadata.get("time"))

        doc = Document(page_content=content, metadata=metadata)
        collection = self._get_collection(memory_type)

        if upsert:
            # LangChain Chroma 没有 upsert_documents，用底层 chromadb Collection
            collection._collection.upsert(
                ids=[memory_id],
                documents=[content],
                metadatas=[metadata],
            )
        else:
            collection.add_documents([doc], ids=[memory_id])

        return memory_id

    # ---------- 2. get_memory ----------

    def get_memory(
        self,
        memory_id: str,
        memory_type: str | None = None,
    ) -> dict | None:
        """
        根据 memory_id 读取一条长期记忆。

        Returns:
            {"memory_id", "content", "metadata", "memory_type"} 或 None
        """
        if memory_type:
            # 只查指定 collection
            result = self._chroma_get_by_id(self._get_collection(memory_type), memory_id)
            if result:
                return self._parse_single(memory_type, result, 0)
            return None

        # 遍历所有 collection
        for mt, col in self._iter_memory_collections():
            result = self._chroma_get_by_id(col, memory_id)
            if result:
                return self._parse_single(mt, result, 0)
        return None

    def _parse_single(self, memory_type: str, chroma_result: dict, index: int) -> dict:
        """将 Chroma 原生 get 结果解析为统一格式。"""
        meta = chroma_result["metadatas"][index] or {}
        if "memory_id" not in meta:
            meta["memory_id"] = chroma_result["ids"][index]
        return {
            "memory_id": meta["memory_id"],
            "content": chroma_result["documents"][index],
            "metadata": meta,
            "memory_type": meta.get("memory_type", memory_type),
        }

    # ---------- 3. update_memory ----------

    def update_memory(
        self,
        memory_id: str,
        metadata: dict | None = None,
        content: str | None = None,
        memory_type: str | None = None,
    ) -> bool:
        """
        更新已有长期记忆。新 metadata 覆盖旧 metadata 对应字段，不会丢失旧字段。

        Returns:
            成功返回 True，找不到返回 False
        """
        # 定位
        old = self.get_memory(memory_id, memory_type=memory_type)
        if old is None:
            return False

        actual_type = old["memory_type"]
        new_meta = dict(old["metadata"])  # 旧值
        if metadata:
            new_meta.update(metadata)

        # 保护不可覆盖字段
        new_meta["memory_id"] = memory_id
        new_meta["memory_type"] = actual_type
        # npc_id 如果旧值存在，不丢失
        if "npc_id" in old["metadata"]:
            new_meta.setdefault("npc_id", old["metadata"]["npc_id"])

        new_content = content if content is not None else old["content"]

        # 用底层 upsert 覆写
        self._ensure_embedding()
        self._get_collection(actual_type)._collection.upsert(
            ids=[memory_id],
            documents=[new_content],
            metadatas=[new_meta],
        )
        return True

    # ---------- 4. query_by_type ----------

    def query_by_type(
        self,
        memory_type: str,
        npc_id: str | None = None,
        query_text: str | None = None,
        top_k: int = 10,
        where: dict | None = None,
        include_archived: bool = False,
    ) -> list[dict]:
        """
        按记忆类型查询长期记忆。

        - 传了 query_text → 向量相似度查询
        - 没传 query_text → metadata get 查询
        - 默认过滤 status == "archived"
        """
        collection = self._get_collection(memory_type)

        # 构建 where 过滤
        conditions = []
        if npc_id:
            conditions.append({"npc_id": {"$eq": npc_id}})
        if where:
            for k, v in where.items():
                conditions.append({k: {"$eq": v}})

        chroma_where = None
        if len(conditions) > 1:
            chroma_where = {"$and": conditions}
        elif conditions:
            chroma_where = conditions[0]

        if query_text:
            # 向量相似度查询
            self._ensure_embedding()
            results = collection._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=chroma_where,
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            items = []
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i] or {}
                if "memory_id" not in meta:
                    meta["memory_id"] = results["ids"][0][i]

                # archived 过滤
                if not include_archived and meta.get("status") == "archived":
                    continue

                items.append({
                    "memory_id": meta["memory_id"],
                    "content": results["documents"][0][i],
                    "metadata": meta,
                    "similarity": 1.0 - results["distances"][0][i],
                })
            return items
        else:
            # metadata get 查询
            results = collection._collection.get(
                where=chroma_where,
                include=["documents", "metadatas"],
                limit=top_k,
            )

            if not results["ids"]:
                return []

            items = []
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i] or {}
                if "memory_id" not in meta:
                    meta["memory_id"] = results["ids"][i]

                if not include_archived and meta.get("status") == "archived":
                    continue

                items.append({
                    "memory_id": meta["memory_id"],
                    "content": results["documents"][i],
                    "metadata": meta,
                    "similarity": None,
                })
            return items

    # ---------- 5. archive_memory ----------

    def archive_memory(
        self,
        memory_id: str,
        memory_type: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """
        软删除：将 status 设为 archived，保留可追溯性。

        Returns:
            成功返回 True，找不到返回 False
        """
        extra_meta = {
            "status": "archived",
            "archived_at": _time.time(),
        }
        if reason:
            extra_meta["archived_reason"] = reason

        return self.update_memory(memory_id, metadata=extra_meta, memory_type=memory_type)

    # ---------- 6. delete_memory ----------

    def delete_memory(
        self,
        memory_id: str,
        memory_type: str | None = None,
        hard: bool = False,
    ) -> bool:
        """
        删除长期记忆。

        - hard=False → 调用 archive_memory (软删除)
        - hard=True  → 物理删除

        Returns:
            成功返回 True，找不到返回 False
        """
        if not hard:
            return self.archive_memory(memory_id, memory_type, reason="soft_delete")

        # 物理删除
        if memory_type:
            # 指定了 collection，直接删
            if memory_type not in self.collections:
                return False
            try:
                self.collections[memory_type]._collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False

        # 未指定 collection，遍历查找后删除
        for mt, col in self._iter_memory_collections():
            result = self._chroma_get_by_id(col, memory_id)
            if result:
                col._collection.delete(ids=[memory_id])
                return True
        return False
