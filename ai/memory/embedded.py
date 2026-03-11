import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil
from typing import List, Dict, Optional
from langchain_core.documents import Document

class MemoryStore:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self._embeddings = None
        # 初始化时 embedding_function 设为 None，等待懒加载
        self.collections = {
            "persona_seed": self._create_collection(db_path, "persona_seed"),
            "event": self._create_collection(db_path, "event"),
            "summary": self._create_collection(db_path, "summary")
        }

    @property
    def embedding_function(self):
        """懒加载模型，并注入到所有 Collection 中"""
        if self._embeddings is None:
            print("--- [Lazy Load] 正在初始化重型模型 SentenceTransformer ---")
            from sentence_transformers import SentenceTransformer
            from langchain_core.embeddings import Embeddings

            class InnerEmbedder(Embeddings):
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)
                def embed_documents(self, texts: List[str]):
                    return self.model.encode(texts, normalize_embeddings=True).tolist()
                def embed_query(self, text: str):
                    return self.model.encode(text, normalize_embeddings=True).tolist()

            self._embeddings = InnerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            
            # 【关键修复】加载完成后，必须手动把 EF 注入回现有的 Chroma 对象
            # 否则调用 add_documents 时会报错 "embedding_function is None"
            for layer_name, collection in self.collections.items():
                collection.embedding_function = self._embeddings

        return self._embeddings

    def _create_collection(self, db_path, collection_name):
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=None # 初始为空
        )
    
    def clear_database(self):
        """物理删除数据库文件夹，彻底重置"""
        if os.path.exists(self.db_path):
            # 必须先关闭所有连接（如果有的话），或者直接删除
            shutil.rmtree(self.db_path)
            print(f"已清理数据库目录: {self.db_path}")
            # 顺便清理标记文件
            if os.path.exists(".persona_init_done"):
                os.remove(".persona_init_done")
    
    def add(self, layer: str, content: str, metadata: Optional[Dict]=None, doc_id: Optional[str]=None):
        # 【关键】在添加前，强制触发一次懒加载，确保 embedding_function 存在
        _ = self.embedding_function
        
        doc = Document(page_content=content, metadata=metadata or {})
        self.collections[layer].add_documents([doc], ids=[doc_id] if doc_id else None)

    def upsert(
        self,
        layer: str,
        content: str,
        metadata:Optional[Dict]=None,
        doc_id:Optional[str]=None 
    ):
        doc = Document(
            page_content=content,
            metadata=metadata or {}
        )

        self.collections[layer].upsert_documents(
            documents=[doc],
            ids=[doc_id] if doc_id else None
        )

    def delete(self, layer: str, ids: List[str]):
        self.collections[layer].delete(ids=ids)

    def similarity_search(
        self,
        layer: str,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        return self.collections[layer].similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
 # memory/embedded.py

    def query(self, layer, filter=None, limit=10):
        # 将简单的字典转换为带操作符的字典
        formatted_filter = None
        if filter:
            # 如果你传入的是 {'npc_id': 'Damon', 'location': 'persona_seed'}
            # 转换为新版要求的 $and 结构
            conditions = []
            for key, value in filter.items():
                conditions.append({key: {"$eq": value}})
            
            if len(conditions) > 1:
                formatted_filter = {"$and": conditions}
            else:
                formatted_filter = conditions[0]

        # 使用转换后的过滤器
        return self.collections[layer].get(where=formatted_filter, limit=limit)
    
    def raw_query(self, layer: str, query_emb: List[float], top_k: int, filter: Optional[Dict] = None):
        """
        直接调用 Chroma 底层的 query 方法，绕过 LangChain 的封装。
        因为我们需要直接传 query_embeddings，而不是 text。
        """
        # LangChain Chroma 对象有一个私有属性 _collection 指向原生的 chromadb.Collection
        chroma_collection = self.collections[layer]._collection
        
        # 转换 filter 格式 (如果是简单的 dict)
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
        """
        获取指定时间范围内的 Event (用于周结)
        """
        # ChromaDB 的 where 过滤语法
        # 注意：metadata 里的 time 必须是 int 或 float
        where_filter = {
            "$and": [
                {"npc_id": {"$eq": npc_id}},
                {"time": {"$gte": start_time}},
                {"time": {"$lte": end_time}},
                {"memory_type": {"$eq": "event"}} # 只拉取具体的事件，不拉取之前的 summary
            ]
        }

        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"] # 我们只需要内容和元数据
        )
        
        # 组装成友好的 list
        events = []
        if results["documents"]:
            for i in range(len(results["documents"])):
                events.append({
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
        
        return events
    
    def get_active_npc_ids(self, start_time: int, end_time: int) -> List[str]:
        """
        获取指定时间段内有过 Event 记录的所有 NPC ID
        """
        # 只要查 metadata 即可，不需要 load document，速度很快
        results = self.collection.get(
            where={
                "$and": [
                    {"time": {"$gte": start_time}},
                    {"time": {"$lte": end_time}},
                    {"type": {"$eq": "raw_event"}} # 只看原始事件
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