from typing import List, Dict, Literal
import math

# =========================
#  记忆字段规范
# =========================
# 每条记忆必须包含以下 10 个字段：
#   memory_id    : str   - 唯一标识
#   memory_type  : str   - persona_seed / episodic_event / preference_belief / relationship_impression / narrative_arc
#   npc_id       : str   - 所属 NPC
#   content      : str   - 记忆文本
#   time         : float | "static" - 发生时间（游戏天数 / "static" 表示固有设定）
#   location     : str   - 发生地点
#   importance   : float - 重要度 [0, 1]
#   status       : "active" | "consolidated" | "dormant" | "archived"
#   last_access  : float - 最近一次被检索的时间
#   source       : str   - 来源（dialogue / observation / reflection / init / ...）
#
# status 说明：
#   active      - 活跃记忆，正常参与检索与衰减
#   consolidated- 已被周结/反思合并，原始事件降级
#   dormant     - 长期未被访问，暂停参与检索（可被重新激活）
#   archived    - 已归档，不再参与检索，仅供审计

MemoryStatus = Literal["active", "consolidated", "dormant", "archived"]

REQUIRED_MEMORY_FIELDS = [
    "memory_id", "memory_type", "npc_id", "content",
    "time", "location", "importance", "status", "last_access", "source",
]

VALID_STATUSES = {"active", "consolidated", "dormant", "archived"}


def validate_memory(memory: Dict) -> bool:
    for k in REQUIRED_MEMORY_FIELDS:
        if k not in memory:
            print(f"[Validator] Missing key '{k}' in memory {memory.get('memory_id', '?')}")
            return False
    status = memory.get("status", "")
    if status not in VALID_STATUSES:
        print(f"[Validator] Invalid status '{status}' in memory {memory.get('memory_id', '?')}")
        return False
    return True

def time_decay(memory_time: float, now_time: float, half_life: float = 10.0) -> float:
    delta = max(0.0, now_time - memory_time)
    return math.exp(-delta / half_life)

#======排序召回======
class MemoryRetriever:
    def __init__(self, memory_store):
        # 这里传入整个 MemoryStore 对象，而不是 vector_db
        self.store = memory_store
        
    @property
    def embed_fn(self):
        # 动态从 store 获取，利用 store 的懒加载特性
        return self.store.embedding_function.embed_query

    # 需要做 sparse hybrid 的层
    _BOOST_LAYERS = {"episodic_event", "persona_seed", "relationship_impression"}

    @property
    def _sparse_model(self):
        """懒加载 FlagEmbedding 模型（共享 embedded.py 的 singleton）。"""
        from memory.embedded import _get_embed_model
        return _get_embed_model()

    @staticmethod
    def _sparse_similarity(query_sparse: dict, doc_sparse_json: str) -> float:
        """计算 query sparse 与文档 sparse 的相似度（weighted dot product）。

        query_sparse: {"token_str": weight, ...}
        doc_sparse_json: JSON string of {token_str: weight}

        返回值已归一化到 [0,1]，由 query 自身范数归一化。
        """
        if not query_sparse or not doc_sparse_json:
            return 0.0
        try:
            import json
            doc_sparse = json.loads(doc_sparse_json)
        except Exception:
            return 0.0

        # query 范数（用于归一化）
        q_norm = sum(w * w for w in query_sparse.values()) ** 0.5
        if q_norm == 0:
            return 0.0

        # 交集加权求和 / query 范数 → 归一化到 [0, 1]
        score = 0.0
        for token_str, q_weight in query_sparse.items():
            d_weight = doc_sparse.get(token_str, 0.0)
            if d_weight > 0:
                score += q_weight * d_weight
        return score / q_norm

    def retrieve(
        self,
        query_text: str,
        npc_id: str,
        persona_k: int,
        episodic_k: int,
        relation_k: int,
    ) -> Dict[str,List[Dict]]:

        # 1. 生成 Query 向量（dense + sparse）
        from memory.embedded import encode_dense_sparse
        query_emb, query_sparse = encode_dense_sparse(query_text)

        # 2. 分层检索（sparse hybrid 覆盖全部三层）
        def _fetch(layer, k):
            if layer in self._BOOST_LAYERS:
                return self._hybrid_retrieve(query_emb, npc_id, layer, k, query_sparse)
            return self._retrieve_by_layer(query_emb, npc_id, layer, k)

        persona = _fetch("persona_seed", persona_k)
        episodic = _fetch("episodic_event", episodic_k)
        relations = _fetch("relationship_impression", relation_k)

        print(f"[Retriever] persona: {len(persona)}, episodic: {len(episodic)}, "
              f"relations: {len(relations)}")
        return {
            "persona_seed": persona,
            "episodic_event": episodic,
            "relationship_impression": relations,
        }

    def _hybrid_retrieve(self, query_emb, npc_id, layer, top_k,
                         query_sparse: dict | None = None):
        """混合检索：dense 向量召回 + BGE-M3 sparse 词汇匹配重排序。

        取 top_k × 2 候选，对每条候选计算 sparse 相似度，与 dense 分数加权合并。
        """
        fetch_k = top_k * 2
        candidates = self._retrieve_by_layer(query_emb, npc_id, layer, fetch_k)

        if not query_sparse or not candidates:
            return candidates[:top_k]

        # dense + sparse 加权权重
        ALPHA_DENSE = 0.7

        for mem in candidates:
            meta = mem.get("metadata", {})
            doc_sparse_json = meta.get("sparse_json", "")
            sparse_score = self._sparse_similarity(query_sparse, doc_sparse_json)
            dense_score = mem.get("similarity", 0.0)
            mem["similarity"] = ALPHA_DENSE * dense_score + (1 - ALPHA_DENSE) * sparse_score
            mem["sparse_score"] = round(sparse_score, 4)

        candidates.sort(key=lambda m: m["similarity"], reverse=True)
        return candidates[:top_k]

    def _retrieve_by_layer(self, query_emb: List[float], npc_id: str, layer: str, top_k: int) -> List[Dict]:
        if top_k <= 0:
            return []

        try:
            # 【关键修改】调用 MemoryStore 新增的 raw_query
            # 注意：这里不需要再传 memory_type 过滤了，因为 collection 已经是独立的了
            results = self.store.raw_query(
                layer=layer,
                query_emb=query_emb,
                top_k=top_k,
                filter={"npc_id": npc_id} 
            )
        except Exception as e:
            print(f"[Retriever] VectorDB Error on layer {layer}: {e}")
            return []

        # 解析 Chroma 原生返回格式
        # Chroma 返回的是 columns 格式：ids=[[id1, id2]], documents=[[doc1, doc2]]
        if not results['ids'] or not results['ids'][0]:
            return []

        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        memories = []
        for i in range(len(docs)):
            # 处理 metadata 可能为 None 的情况
            meta = metas[i] if metas[i] else {}
            
            # 补全 id (防止 meta 里没存 memory_id)
            if "memory_id" not in meta:
                meta["memory_id"] = ids[i]

            mem = {
                "memory_id": meta["memory_id"],
                "content": docs[i],
                "similarity": 1 - distances[i], # 距离越小越相似，简单转分
                "metadata": meta
            }
            # 验证（需引入 validate_memory）
            # if validate_memory(mem["metadata"]): ...
            memories.append(mem)

        return memories

class MemoryRanker:
    def __init__(
        self,
        similarity_weight: float = 1.0,
        importance_weight: float = 1.0,
        recency_weight: float = 1.0,
        location_weight: float = 0.5,
        half_life: float = 10.0,
        layer_weights: Dict[str,float] = None
    ):
        self.sim_w = similarity_weight
        self.imp_w = importance_weight
        self.rec_w = recency_weight
        self.loc_w = location_weight
        self.half_life = half_life
        self.layer_weights = layer_weights or {
            "persona_seed": 1.2,
            "episodic_event": 0.6,
            "relationship_impression": 1.1,
        }

    # 场景标签匹配加分权重
    SCENE_TAG_BONUS = 0.25

    def score(self, memory: Dict, now_time: float, current_loc: str,
              scene_tags: set = None) -> float:
        meta = memory["metadata"]

        # 0. status 过滤
        status = meta.get("status", "active")
        if status == "archived":
            return -1.0  # 彻底退役，等待物理删除

        # dormant 可被检索但重罚，只有高度相关且无 active 覆盖时才浮上来
        dormant_penalty = 0.3 if status == "dormant" else 1.0

        # 1. 基础分：相似度 + 重要性
        sim = memory.get("similarity", 0.0)
        imp = meta.get("importance", 0.5)

        # consolidated 记忆额外降权（已被摘要吸收）
        consolidated_penalty = 0.6 if status == "consolidated" else 1.0

        # 2. 时间衰减
        mem_time = meta.get("time")
        if mem_time == "static" or not isinstance(mem_time, (int, float)):
            decay = 1.0
        else:
            delta = max(0.0, now_time - mem_time)
            decay = math.exp(-delta / self.half_life)

        # 3. 空间加成
        mem_loc = meta.get("location", "")
        loc_bonus = 1.0 if (mem_loc and mem_loc == current_loc) else 0.0

        # 4. 场景标签加成（新增）
        # 如果记忆的 topic_tags / season / weather 与当前场景匹配，加分
        tag_bonus = 0.0
        if scene_tags:
            topic_tags = set(meta.get("topic_tags", []))
            if isinstance(meta.get("topic_tags"), str):
                try:
                    import json
                    topic_tags = set(json.loads(meta["topic_tags"]))
                except Exception:
                    topic_tags = set()
            mem_season = str(meta.get("season", "")).lower()
            mem_weather = str(meta.get("weather", "")).lower()
            if (topic_tags & scene_tags or
                mem_season in scene_tags or
                mem_weather in scene_tags):
                tag_bonus = self.__class__.SCENE_TAG_BONUS

        # 5. 层级系数
        layer_w = self.layer_weights.get(meta.get("memory_type", "episodic_event"), 1.0)

        # 综合打分公式
        raw_score = (
            sim * self.sim_w +
            imp * self.imp_w +
            decay * self.rec_w +
            loc_bonus * self.loc_w +
            tag_bonus
        )

        return raw_score * layer_w * consolidated_penalty * dormant_penalty

    def rank(self, memories: List[Dict], now_time: float, current_loc: str = "Unknown",
             top_k: int = 5, scene_tags: set = None) -> List[Dict]:
        if not memories:
            return []

        scored = []
        for m in memories:
            m["final_score"] = self.score(m, now_time, current_loc, scene_tags)
            scored.append(m)

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        # 调试打印
        print(f"[Ranker] Top {min(len(scored), 3)}:")
        for m in scored[:3]:
             print(f"  [{m['metadata'].get('memory_type')}] Score: {m['final_score']:.3f} "
                   f"| Content: {m['content'][:30]}...")

        return scored[:top_k]

# =========================
# Memory WriteBack
# =========================
class MemoryWriteBack:
    def __init__(self, reinforce_rate: float = 0.03, cold_decay: float = 0.995,
                 dormant_threshold: float = 0.1, archive_threshold: float = 0.03):
        self.reinforce_rate = reinforce_rate
        self.cold_decay = cold_decay
        self.dormant_threshold = dormant_threshold
        self.archive_threshold = archive_threshold

    def update(self, all_candidates: List[Dict], used_memories: List[Dict], now_time: float):
        used_ids = {m["memory_id"] for m in used_memories}

        for m in all_candidates:
            status = m.get("status", "active")

            # 已归档的记忆不再变动
            if status == "archived":
                continue

            if m["memory_id"] in used_ids:
                # 被检索到 → 强化
                m["last_access"] = now_time
                score = m.get("self_rag_score", 0.0)
                m["importance"] = min(1.0, m.get("importance", 0.5) + self.reinforce_rate * score)
                # dormant → 重新激活
                if status == "dormant":
                    m["status"] = "active"
                    print(f"[WriteBack] Reactivated {m['memory_id']}")
                print(f"[WriteBack] Reinforced {m['memory_id']} importance={m['importance']:.3f}")
            else:
                # 未被检索 → 衰减
                m["importance"] = m.get("importance", 0.5) * self.cold_decay

                # 根据 importance 阈值自动降级 status
                if status == "active" and m["importance"] < self.dormant_threshold:
                    m["status"] = "dormant"
                    print(f"[WriteBack] Dormant {m['memory_id']} importance={m['importance']:.3f}")
                elif status == "dormant" and m["importance"] < self.archive_threshold:
                    m["status"] = "archived"
                    print(f"[WriteBack] Archived {m['memory_id']} importance={m['importance']:.3f}")
                else:
                    print(f"[WriteBack] Cooled {m['memory_id']} importance={m['importance']:.3f}")

#======= self-rag=======
#TODO

#=======接口=======
class ContextManager:
    def __init__(self, retriever: MemoryRetriever, ranker: MemoryRanker):
        self.retriever = retriever
        self.ranker = ranker

    def get_context(self, rel_desc:str,rel_instr:str,player_input: str, state: Dict, return_raw: bool = False):
        rich_query = self._build_rich_query(rel_desc,rel_instr,player_input, state)

        print(f"[ContextManager] Rich Query: {rich_query}",flush=True)

        current_time = state.get("time_num", 1)
        current_loc = state.get("location", "Unknown")
        relationship = state.get("relationship", "")

        # 提取场景标签，传给 ranker 做 tag bonus
        scene_tags = set()
        for field in ("season", "weather"):
            val = str(state.get(field, "")).lower()
            if val and val != "any":
                scene_tags.add(val)
        # relationship stage 也是关键场景信息
        if relationship:
            scene_tags.add(relationship.lower())

        raw_results = self.retriever.retrieve(
            query_text=rich_query,
            npc_id=state.get("npc_id", "Damon"),
            persona_k=3,
            episodic_k=5,
            relation_k=1,
        )

        final_persona = self.ranker.rank(raw_results["persona_seed"], current_time, current_loc, top_k=3, scene_tags=scene_tags)
        final_episodic = self.ranker.rank(raw_results["episodic_event"], current_time, current_loc, top_k=5, scene_tags=scene_tags)
        final_relations = self.ranker.rank(raw_results["relationship_impression"], current_time, current_loc, top_k=1, scene_tags=scene_tags)

        # 更新 last_access
        for mem in final_episodic + final_relations:
            meta = mem.get("metadata", {})
            update_fields = {"last_access": current_time}
            if meta.get("status") == "dormant":
                update_fields["status"] = "active"
            try:
                self.retriever.store.update_memory(
                    mem["memory_id"],
                    metadata=update_fields,
                )
            except Exception:
                pass  # 静默失败，不影响主流程

        episodic_text = self._format_list(final_episodic, "episodic_event", current_time)
        relation_text = self._format_list(final_relations, "relationship_impression", current_time)

        event_text = episodic_text if episodic_text else "（无相关记录）"
        summary_text = relation_text if relation_text else "（无相关记录）"

        formatted = {
            "persona_text": self._format_list(final_persona, "persona_seed", current_time),
            "event_text": event_text,
            "summary_text": summary_text,
        }

        if return_raw:
            return formatted, raw_results
        return formatted

    def _build_rich_query(self, rel_desc:str,rel_instr:str,player_input: str, state: Dict) -> str:
        """构造检索 query。

        设计原则：
        - player_input 前置，给予最大向量权重
        - rel_desc 截断到首句（长描述会稀释信号）
        - 显式注入关系阶段标签，帮助 embedding 模型理解场景
        - rel_instr 是 LLM 行为指令，对检索无用，不加入 query
        """
        relationship = state.get("relationship", "")
        season = state.get("season", "")
        weather = state.get("weather", "")
        location = state.get("location", "")

        # 截断长描述（保留首句，去掉冗长的行为细则）
        if len(rel_desc) > 60:
            cut = rel_desc.find("。", 0, 60)
            rel_desc = rel_desc[:cut] if cut > 0 else rel_desc[:60]

        # 构造 query：玩家输入最前，场景标签居中，NPC 视角收尾
        parts = [f"玩家说: {player_input}"]

        scene_tags = []
        if relationship:
            scene_tags.append(f"关系: {relationship}")
        if season:
            scene_tags.append(f"季节: {season}")
        if weather:
            scene_tags.append(f"天气: {weather}")
        if location:
            scene_tags.append(f"地点: {location}")
        if scene_tags:
            parts.append("[" + ", ".join(scene_tags) + "]")

        if rel_desc:
            parts.append(f"NPC视角: {rel_desc}")

        return "。".join(parts)
    
    def _format_list(self, memories: List[Dict], layer: str, current_time: float = 0) -> str:
        if not memories:
            return "（无相关记录）"
        
        lines = []
        for m in memories:
            content = m["content"]
            if layer == "persona_seed":
                lines.append(f"- {content}")
            elif layer == "episodic_event":
                meta = m["metadata"]
                loc = meta.get("location", "未知")
                raw_time = meta.get("time")
                time_str = self._get_relative_time(raw_time, current_time)
                lines.append(f"- [{time_str} @ {loc}] {content}")
            elif layer == "relationship_impression":
                meta = m["metadata"]
                target = meta.get("target", "")
                prefix = f"[→{target}] " if target else ""
                lines.append(f"- {prefix}{content}")
            else:
                lines.append(f"- {content}")
                
        return "\n".join(lines)
    def _get_relative_time(self, mem_time, current_time) -> str:
        if mem_time == "static" or mem_time is None:
            return "固有设定"
            
        try:
            delta = float(current_time) - float(mem_time)
        except (ValueError, TypeError):
            return "未知时间"

        if delta < 2:
            return "昨天"
        elif delta < 7:
            return f"{int(delta)}天前"
        elif delta <30:
            return "不到一个月时"
        elif delta <100:
            return "几个月前"
        else:
            return "很久以前"