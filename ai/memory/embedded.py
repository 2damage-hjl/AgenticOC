import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 离线优先：如果模型已在 HF cache 中，不走网络
if os.environ.get("TRANSFORMERS_OFFLINE") is None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import shutil
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Optional

import lancedb

# ---- BGE-M3 singleton (shared with few-shot pipeline) ----

_embed_model = None

def _resolve_local_path(model_name: str = "BAAI/bge-m3") -> str | None:
    """在 HF cache 中查找 BGE-M3 的本地 snapshot 路径。"""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = "models--" + model_name.replace("/", "--")
    model_dir = cache_root / folder_name / "snapshots"
    if not model_dir.is_dir():
        return None
    # 找最新的 snapshot
    candidates = sorted(
        [d for d in model_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snap in candidates:
        # FlagEmbedding 需要 config.json + tokenizer
        if (snap / "config.json").exists() and (snap / "tokenizer.json").exists():
            return str(snap)
    return None


def _get_embed_model():
    """Lazy-load BGE-M3 via FlagEmbedding, exposes dense + sparse."""
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import BGEM3FlagModel
        local_path = os.environ.get("BGE_M3_LOCAL_PATH") or _resolve_local_path()
        model_ref = local_path or "BAAI/bge-m3"
        _embed_model = BGEM3FlagModel(model_ref, use_fp16=True)
    return _embed_model


def encode_dense(text: str):
    """BGE-M3 dense vector only（兼容旧接口）。"""
    return _get_embed_model().encode(
        text, return_dense=True, return_sparse=False
    )['dense_vecs'].tolist()


def encode_dense_sparse(text: str):
    """BGE-M3 dense + sparse 双输出。

    Returns:
        (dense_vec: List[float], lexical_weights: Dict[str, float])
        lexical_weights 的 key 是 token 字符串（与 BGE-M3 tokenizer 对齐）。
    """
    out = _get_embed_model().encode(
        text, return_dense=True, return_sparse=True
    )
    dense = out['dense_vecs'].tolist()
    # ponytail: numpy float32 → native float, 否则 json.dumps 炸
    sparse = {k: float(v) for k, v in out['lexical_weights'].items()}
    return dense, sparse


def _get_data_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class MemoryStore:
    MEMORY_TYPES = [
        "persona_seed",
        "episodic_event",
        "relationship_impression",
    ]

    _ID_PREFIX = {
        "persona_seed": "persona",
        "episodic_event": "evt",
        "relationship_impression": "rel",
    }

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(_get_data_dir(), "memory_lancedb")
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self._tables = {}
        self._embed_fn = None

    # =========================
    #  embedding (shared BGE-M3)
    # =========================

    @property
    def embedding_function(self):
        if self._embed_fn is None:
            class _Wrapper:
                def embed_query(self_, text: str):
                    return encode_dense(text)
                def embed_documents(self_, texts: List[str]):
                    return [encode_dense(t) for t in texts]

            self._embed_fn = _Wrapper()
        return self._embed_fn

    def _encode(self, text: str) -> List[float]:
        return encode_dense(text)

    # =========================
    #  table management
    # =========================

    @property
    def collections(self):
        """兼容旧接口：返回 {memory_type: _TableWrapper} dict."""
        for mt in self.MEMORY_TYPES:
            if mt not in self._tables:
                self._tables[mt] = self._open_or_none(mt)
        return self._tables

    def _get_collection(self, memory_type: str):
        """兼容旧接口：返回一个包含 _collection 属性的 wrapper，支持 ._collection.count()。"""
        tbl = self._open_or_none(memory_type)
        class _Wrapper:
            def __init__(_s, table): _s._collection = _CollectionWrapper(table)
        class _CollectionWrapper:
            def __init__(_s, table): _s._table = table
            def count(_s):
                if _s._table is None:
                    return 0
                try:
                    return _s._table.count_rows()
                except Exception:
                    return 0
        return _Wrapper(tbl)

    def _open_or_none(self, memory_type: str):
        """尝试打开已存在的 table，不存在则返回 None。"""
        name = f"memory_{memory_type}"
        try:
            return self.db.open_table(name)
        except Exception:
            return None

    def _get_table(self, memory_type: str):
        """获取已存在的 table，不存在则抛异常。"""
        if memory_type not in self.MEMORY_TYPES:
            raise KeyError(f"未知 memory_type: '{memory_type}'，可用: {self.MEMORY_TYPES}")
        if memory_type not in self._tables:
            self._tables[memory_type] = self._open_or_none(memory_type)
        tbl = self._tables[memory_type]
        if tbl is None:
            raise KeyError(f"Table 'memory_{memory_type}' 尚未创建")
        return tbl

    def _get_or_create_table(self, memory_type: str, first_record: dict):
        """获取 table，如果不存在则用第一条记录创建。"""
        name = f"memory_{memory_type}"
        tbl = self._open_or_none(memory_type)
        if tbl is None:
            tbl = self.db.create_table(name, [first_record])
        self._tables[memory_type] = tbl
        return tbl

    def _iter_memory_collections(self):
        for mt in self.MEMORY_TYPES:
            tbl = self._open_or_none(mt)
            if tbl is not None:
                yield mt, tbl

    @classmethod
    def _auto_id(cls, memory_type: str) -> str:
        prefix = cls._ID_PREFIX.get(memory_type, "mem")
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    # =========================
    #  内部工具
    # =========================

    @staticmethod
    def _parse_row(r: dict) -> dict:
        """解析 LanceDB 行，把 metadata_json + sparse_json 展开为 metadata dict。"""
        import json as _json
        meta = {}
        if "metadata_json" in r and r["metadata_json"]:
            try:
                meta = _json.loads(r["metadata_json"])
            except Exception:
                pass
        if "sparse_json" in r and r["sparse_json"]:
            meta["sparse_json"] = r["sparse_json"]
        # 兼容旧格式
        for k, v in r.items():
            if k in ("content", "memory_id", "vector", "_distance",
                     "metadata_json", "sparse_json", "npc_id"):
                continue
            meta.setdefault(k, v)
        return meta

    # =========================
    #  兼容旧接口
    # =========================

    def clear_database(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"已清理数据库目录: {self.db_path}")
            self._tables = {}
            self.db = lancedb.connect(self.db_path)

    def add(self, layer: str, content: str, metadata: Optional[Dict] = None,
            doc_id: Optional[str] = None):
        return self.add_memory(memory_type=layer, content=content,
                               metadata=metadata or {}, memory_id=doc_id)

    def upsert(self, layer: str, content: str, metadata: Optional[Dict] = None,
               doc_id: Optional[str] = None):
        return self.add_memory(memory_type=layer, content=content,
                               metadata=metadata or {}, memory_id=doc_id, upsert=True)

    def delete(self, layer: str, ids: List[str]):
        tbl = self._open_or_none(layer)
        if tbl is not None:
            for mid in ids:
                try:
                    tbl.delete(f"memory_id = '{mid}'")
                except Exception:
                    pass

    def similarity_search(self, layer: str, query: str, k: int = 5,
                          filter: Optional[Dict] = None):
        from langchain_core.documents import Document
        vec = self._encode(query)
        results = self._raw_query_inner(layer, vec, k, filter)
        docs = []
        for i in range(len(results.get("ids", [[]])[0])):
            docs.append(Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            ))
        return docs

    # =========================
    #  核心方法
    # =========================

    def raw_query(self, layer: str, query_emb: List[float], top_k: int,
                  filter: Optional[Dict] = None):
        """向量检索，返回 ChromaDB 兼容格式。"""
        return self._raw_query_inner(layer, query_emb, top_k, filter)

    def _raw_query_inner(self, layer: str, query_emb: List[float], top_k: int,
                         filter: Optional[Dict] = None) -> dict:
        """实际向量检索逻辑。

        返回 ChromaDB 兼容格式:
            {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        """
        tbl = self._open_or_none(layer)
        if tbl is None:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        builder = tbl.search(query_emb).metric("cosine").limit(top_k)
        if filter:
            parts = []
            for k, v in filter.items():
                if isinstance(v, str):
                    # LanceDB SQL: 单引号表示字符串字面量，双引号是标识符
                    escaped = v.replace("'", "''")
                    parts.append(f"{k} = '{escaped}'")
                elif isinstance(v, (int, float)):
                    parts.append(f'{k} = {v}')
            if parts:
                builder = builder.where(" AND ".join(parts))

        try:
            rows = builder.to_list()
        except Exception:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        ids, docs, metas, dists = [], [], [], []
        for r in rows:
            ids.append(r.get("memory_id", ""))
            docs.append(r.get("content", ""))
            metas.append(self._parse_row(r))
            dists.append(1.0 - r.get("_distance", 0.0))

        return {
            'ids': [ids],
            'documents': [docs],
            'metadatas': [metas],
            'distances': [dists],
        }

    def query(self, layer, filter=None, limit=10):
        """元数据查询（无向量搜索）。"""
        tbl = self._open_or_none(layer)
        if tbl is None:
            return {'ids': [], 'documents': [], 'metadatas': []}

        builder = tbl.search().limit(limit)
        if filter:
            parts = []
            for k, v in filter.items():
                if isinstance(v, str):
                    escaped = v.replace("'", "''")
                    parts.append(f"{k} = '{escaped}'")
                else:
                    parts.append(f'{k} = {v}')
            if parts:
                builder = builder.where(" AND ".join(parts))

        try:
            rows = builder.to_list()
        except Exception:
            return {'ids': [], 'documents': [], 'metadatas': []}

        return {
            'ids': [r.get("memory_id", "") for r in rows],
            'documents': [r.get("content", "") for r in rows],
            'metadatas': [self._parse_row(r) for r in rows],
        }

    def get_memory(self, memory_id: str, memory_type: str | None = None) -> dict | None:
        """按 memory_id 读取一条记忆。"""
        types = [memory_type] if memory_type else self.MEMORY_TYPES
        for mt in types:
            tbl = self._open_or_none(mt)
            if tbl is None:
                continue
            try:
                rows = tbl.search().where(f"memory_id = '{memory_id}'").limit(1).to_list()
                if rows:
                    r = rows[0]
                    meta = self._parse_row(r)
                    return {
                        "memory_id": r.get("memory_id", ""),
                        "content": r.get("content", ""),
                        "metadata": meta,
                        "memory_type": meta.get("memory_type", mt),
                    }
            except Exception:
                continue
        return None

    def add_memory(self, memory_type: str, content: str, metadata: dict,
                   memory_id: str | None = None, upsert: bool = False) -> str:
        """新增一条长期记忆。

        核心字段（memory_id, npc_id, content, vector）单独存为列，
        其余 metadata 序列化为 metadata_json，避免 LanceDB 固定 schema 限制。
        """
        import json as _json

        if not memory_id:
            memory_id = self._auto_id(memory_type)

        meta = dict(metadata)
        npc_id = meta.get("npc_id", "")
        meta["memory_id"] = memory_id
        meta["memory_type"] = memory_type
        meta.setdefault("status", "active")
        meta.setdefault("last_access", meta.get("time", 0))

        vec, sparse = encode_dense_sparse(content)
        # 存储 token→weight 映射为 JSON
        sparse_json = _json.dumps(sparse, ensure_ascii=False)

        record = {
            "memory_id": memory_id,
            "npc_id": npc_id,
            "content": content,
            "vector": vec,
            "sparse_json": sparse_json,
            "metadata_json": _json.dumps(meta, ensure_ascii=False),
        }

        if upsert:
            tbl = self._open_or_none(memory_type)
            if tbl is not None:
                try:
                    tbl.delete(f"memory_id = '{memory_id}'")
                except Exception:
                    pass
            self._get_or_create_table(memory_type, record)
            tbl = self._get_table(memory_type)
            tbl.add([record])
        else:
            self._get_or_create_table(memory_type, record)
            tbl = self._get_table(memory_type)
            try:
                existing = tbl.search().where(f"memory_id = '{memory_id}'").limit(1).to_list()
                if not existing:
                    tbl.add([record])
            except Exception:
                tbl.add([record])

        return memory_id

    def update_memory(self, memory_id: str, metadata: dict | None = None,
                      content: str | None = None, memory_type: str | None = None) -> bool:
        """更新已有记忆的 metadata 或 content。

        由于 metadata 存在 metadata_json 中，需要用 delete + add 的方式更新。
        """
        import json as _json

        old = self.get_memory(memory_id, memory_type=memory_type)
        if old is None:
            return False

        actual_type = old["memory_type"]
        tbl = self._get_table(actual_type)

        # 合并 metadata
        new_meta = dict(old["metadata"])
        if metadata:
            new_meta.update(metadata)
        npc_id = new_meta.get("npc_id", "")

        new_content = content if content is not None else old["content"]

        # 删除旧记录，重新添加
        try:
            tbl.delete(f"memory_id = '{memory_id}'")
        except Exception:
            pass

        vec, sparse = encode_dense_sparse(new_content)
        record = {
            "memory_id": memory_id,
            "npc_id": npc_id,
            "content": new_content,
            "vector": vec,
            "sparse_json": _json.dumps(sparse, ensure_ascii=False),
            "metadata_json": _json.dumps(new_meta, ensure_ascii=False),
        }
        tbl.add([record])
        return True

    def query_by_type(self, memory_type: str, npc_id: str | None = None,
                      query_text: str | None = None, top_k: int = 10,
                      where: dict | None = None, include_archived: bool = False) -> list[dict]:
        """按类型查询，支持向量搜索或元数据过滤。"""
        tbl = self._open_or_none(memory_type)
        if tbl is None:
            return []

        if query_text:
            vec = self._encode(query_text)
            builder = tbl.search(vec).metric("cosine").limit(top_k)
        else:
            builder = tbl.search().limit(top_k)

        # where clause
        conditions = []
        if npc_id:
            conditions.append(f'npc_id = "{npc_id}"')
        if not include_archived:
            conditions.append("status != 'archived'")
        if where:
            for k, v in where.items():
                if isinstance(v, str):
                    escaped = v.replace("'", "''")
                    conditions.append(f"{k} = '{escaped}'")
                elif isinstance(v, (int, float)):
                    conditions.append(f'{k} = {v}')

        if conditions:
            builder = builder.where(" AND ".join(conditions))

        try:
            rows = builder.to_list()
        except Exception:
            return []

        results = []
        for r in rows:
            meta = self._parse_row(r)
            results.append({
                "memory_id": r.get("memory_id", ""),
                "content": r.get("content", ""),
                "metadata": meta,
                "memory_type": meta.get("memory_type", memory_type),
            })
        return results

    def get_events_in_time_range(self, npc_id: str, start_time: int, end_time: int) -> List[Dict]:
        """获取时间范围内的 EpisodicEvent。

        time 字段存在 metadata_json 中，LanceDB 无法直接做范围过滤。
        退化为全量扫描 + Python 过滤。
        """
        tbl = self._open_or_none("episodic_event")
        if tbl is None:
            return []

        try:
            rows = tbl.search().where(f"npc_id = '{npc_id}'").limit(1000).to_list()
        except Exception:
            return []

        results = []
        for r in rows:
            meta = self._parse_row(r)
            t = meta.get("time", 0)
            if isinstance(t, str):  # "static" 等
                continue
            try:
                t = float(t)
            except (ValueError, TypeError):
                continue
            if start_time <= t <= end_time:
                results.append({
                    "content": r.get("content", ""),
                    "metadata": meta,
                })
        return results

    def get_active_npc_ids(self, start_time: int, end_time: int) -> List[str]:
        """获取时间范围内有事件的 NPC ID 列表。"""
        tbl = self._open_or_none("episodic_event")
        if tbl is None:
            return []

        try:
            rows = tbl.search().limit(1000).to_list()
        except Exception:
            return []

        ids = set()
        for r in rows:
            npc_id = r.get("npc_id", "")
            if npc_id:
                meta = self._parse_row(r)
                t = meta.get("time", 0)
                if isinstance(t, str):
                    continue
                try:
                    t = float(t)
                except (ValueError, TypeError):
                    continue
                if start_time <= t <= end_time:
                    ids.add(npc_id)
        return list(ids)
