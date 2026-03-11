from typing import List, Dict
import math

def validate_memory(memory: Dict) -> bool:
    required_keys = ["memory_id","npc_id","content","time","last_access","location","memory_type","importance"]
    for k in required_keys:
        if k not in memory:
            print(f"[Validator] Missing key {k} in memory {memory.get('memory_id','?')}")
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

    def retrieve(
        self,
        query_text: str,
        npc_id: str,
        persona_k: int,
        event_k: int,
        summary_k: int 
    ) -> Dict[str,List[Dict]]:
        
        # 1. 生成 Query 向量
        query_emb = self.embed_fn(query_text)

        # 2. 分层去对应的 Collection 查
        persona = self._retrieve_by_layer(query_emb, npc_id, "persona_seed", persona_k)
        events = self._retrieve_by_layer(query_emb, npc_id, "event", event_k)
        summaries = self._retrieve_by_layer(query_emb, npc_id, "summary", summary_k)

        print(f"[Retriever] persona: {len(persona)}, events: {len(events)}, summaries: {len(summaries)}")
        return {
            "persona_seed": persona,
            "event": events,
            "summary": summaries
        }

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
        location_weight: float = 0.5, # 新增：地点加权
        half_life: float = 10.0,
        layer_weights: Dict[str,float] = None
    ):
        self.sim_w = similarity_weight
        self.imp_w = importance_weight
        self.rec_w = recency_weight
        self.loc_w = location_weight # 新增
        self.half_life = half_life
        self.layer_weights = layer_weights or {
            "persona_seed": 1.2, # 人设稍微加权，保证性格稳定
            "summary": 0.8,
            "event": 0.6
        }

    def score(self, memory: Dict, now_time: float, current_loc: str) -> float:
        meta = memory["metadata"]
        
        # 1. 基础分：相似度 + 重要性
        sim = memory.get("similarity", 0.0)
        imp = meta.get("importance", 0.5)
        
        # 2. 时间衰减 (修复：处理 "static" 时间)
        mem_time = meta.get("time")
        if mem_time == "static" or not isinstance(mem_time, (int, float)):
            decay = 1.0 # 静态记忆（如人设）永不衰减
        else:
            delta = max(0.0, now_time - mem_time)
            decay = math.exp(-delta / self.half_life)
            
        # 3. 空间加成 (新增)
        # 如果记忆发生的地点 == 当前地点，给予加分
        mem_loc = meta.get("location", "")
        loc_bonus = 1.0 if (mem_loc and mem_loc == current_loc) else 0.0
        
        # 4. 层级系数
        layer_w = self.layer_weights.get(meta.get("memory_type", "event"), 1.0)

        # 综合打分公式
        # (相似度 + 重要性 + 时间衰减 + 地点加成) * 层级权重
        raw_score = (
            sim * self.sim_w + 
            imp * self.imp_w + 
            decay * self.rec_w + 
            loc_bonus * self.loc_w
        )
        
        return raw_score * layer_w

    # 注意：rank 方法签名变了，需要接收 current_loc
    def rank(self, memories: List[Dict], now_time: float, current_loc: str = "Unknown", top_k: int = 5) -> List[Dict]:
        if not memories:
            return []
            
        scored = []
        for m in memories:
            m["final_score"] = self.score(m, now_time, current_loc)
            scored.append(m)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 调试打印
        print(f"[Ranker] Top {min(len(scored), 3)}:")
        for m in scored[:3]:
             print(f"  [{m['metadata'].get('memory_type')}] Score: {m['final_score']:.3f} | Content: {m['content'][:20]}...")

        return scored[:top_k]

# =========================
# Memory WriteBack
# =========================
class MemoryWriteBack:
    def __init__(self, reinforce_rate: float = 0.03, cold_decay: float = 0.995):
        self.reinforce_rate = reinforce_rate
        self.cold_decay = cold_decay

    def update(self, all_candidates: List[Dict], used_memories: List[Dict], now_time: float):
        used_ids = {m["memory_id"] for m in used_memories}

        for m in all_candidates:
            if m["memory_id"] in used_ids:
                m["last_access"] = now_time
                score = m.get("self_rag_score", 0.0)
                m["importance"] = min(1.0, m.get("importance",0.5) + self.reinforce_rate * score)
                print(f"[WriteBack] Reinforced {m['memory_id']} importance={m['importance']:.3f}")
            else:
                m["importance"] = m.get("importance",0.5) * self.cold_decay
                print(f"[WriteBack] Cooled {m['memory_id']} importance={m['importance']:.3f}")

#======= self-rag=======
#TODO

#=======接口=======
class ContextManager:
    def __init__(self, retriever: MemoryRetriever, ranker: MemoryRanker):
        self.retriever = retriever
        self.ranker = ranker

    def get_context(self, rel_desc:str,rel_instr:str,player_input: str, state: Dict) -> Dict[str, str]:
        rich_query = self._build_rich_query(rel_desc,rel_instr,player_input, state)
        
        print(f"[ContextManager] Rich Query: {rich_query}",flush=True)

        current_time = state.get("time_num", 1)
        current_loc = state.get("location", "Unknown")

        raw_results = self.retriever.retrieve(
            query_text=rich_query, 
            npc_id=state.get("npc_id", "Damon"),
            persona_k=5, 
            summary_k=3, 
            event_k=8 
        )

        final_persona = self.ranker.rank(raw_results["persona_seed"], current_time, current_loc, top_k=3)
        final_summary = self.ranker.rank(raw_results["summary"], current_time, current_loc, top_k=2)
        final_events = self.ranker.rank(raw_results["event"], current_time, current_loc, top_k=5)

        return {
            "persona_text": self._format_list(final_persona, "persona", current_time),
            "summary_text": self._format_list(final_summary, "summary", current_time),
            "event_text":   self._format_list(final_events, "event", current_time)
        }

    def _build_rich_query(self, rel_desc:str,rel_instr:str,player_input: str, state: Dict) -> str:             
        last_npc_reply = state.get("npc_reply", "")
        context_str = f"你刚才说了: '{last_npc_reply[:30]}...'" if last_npc_reply else "话题开启"

        rich_query = (
            f"当前关系: {rel_desc}。 "
            f"社交准则: {rel_instr}。 "
            f"上下文: {context_str}。 "
            f"玩家说: {player_input}"
        )
        
        return rich_query
    
    def _format_list(self, memories: List[Dict], layer: str, current_time: float = 0) -> str:
        if not memories:
            return "（无相关记录）"
        
        lines = []
        for m in memories:
            content = m["content"]
            if layer == "persona_seed":
                lines.append(f"- {content}")
            elif layer == "event":
                meta = m["metadata"]
                loc = meta.get("location", "未知")
                raw_time = meta.get("time")
                
                time_str = self._get_relative_time(raw_time, current_time)
                lines.append(f"- [{time_str} @ {loc}] {content}")
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