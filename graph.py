from langgraph.graph import StateGraph, END
import os
from typing import List, TypedDict, Literal, Optional, Dict
from memory.STMemory import ChatMemory
from llm import load_config
from pprint import pprint

class DialogueState(TypedDict):
    # ===== 输入 / 控制 =====
    command: Literal["NORMAL", "END_DIALOGUE", "CANCEL_DIALOGUE"]
    npc_id: str
    game_time: str
    time_num: int
    location: str
    attitude: str
    relationship: str
    weather: str
    player_info: str
    today_actions: List[str]
    luckystatus: str

    # --- Few-shot 状态 ---
    is_birthday: bool
    is_festival: bool
    is_gifting: bool
    route: str
    game_flags: List[str]
  
    # ===== 对话 =====
    last_user_input: Optional[str]
    npc_reply: Optional[str]

    # ===== Graph 内部状态 =====
    error: Optional[str] 


graph = StateGraph(DialogueState)

from memory.embedded import MemoryStore
from memory.LTMemory import ContextManager,MemoryRetriever, MemoryRanker
# 1. === 全局初始化区 ===
print("系统初始化中...")
store = MemoryStore()             # 加载数据库连接
print(">>> 正在加载 SentenceTransformer 模型 (预热中)...")
_ = store.embedding_function 
print(">>> 长期记忆模型加载完毕")
# Also pre-load the BGE-M3 model for LanceDB few-shot retrieval
print(">>> 正在预热 LanceDB 嵌入模型 (BGE-M3)...")
try:
    from prompt_construction.retrieval.query_lancedb import get_embedding_model
    get_embedding_model()  # loads once, cached for all subsequent calls
    print(">>> BGE-M3 模型加载完毕！(后续 few-shot 检索不再卡顿)")
except Exception as e:
    print(f">>> BGE-M3 预热失败 (few-shot 将 fallback 到静态示例): {e}")
print(">>> 模型加载完毕！(后续对话将不再卡顿)")
retriever = MemoryRetriever(store)# 绑定检索器
ranker = MemoryRanker()           # 绑定排序器
global_context_manager = ContextManager(retriever, ranker) 
print("系统初始化完成！")
from llm import create_llm
llm = create_llm()

import sys

def _get_data_dir() -> str:
    """获取数据存储目录（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

INIT_FLAG = os.path.join(_get_data_dir(), ".persona_init_done")
from memory.scripts.persona_seed import initial_persona_seed, build_damon_persona_seed
from memory.scripts.citizen_seed import init_all_citizens
if not os.path.exists(INIT_FLAG):
    print("--- 正在进行首次人设初始化 (写入向量库) ---")
    try:
        initial_persona_seed(
            npc_id="Damon",
            memory_store=store,
            build_func=build_damon_persona_seed
        )
        init_all_citizens(store)

        # 标记完成
        with open(INIT_FLAG, "w") as f:
            f.write("done")
        print("--- 人设初始化完成 ---")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
else:
    print("检测到人设已初始化，跳过写入。")

print("系统启动完成！等待命令...")

#=======节点处理函数=======

def handle_normal(state: Dict) -> Dict:
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" [DEBUG] 正在检查传入 handle_normal 的 State:")
    pprint(state)
    print("="*50 + "\n")
    # ---------------------------------------------------------
    import time as _time
    from prompt_construction.utils.dialogue_trace import DialogueTrace, StageTimer

    trace: DialogueTrace = state.get("_trace", DialogueTrace())

    player_input = state.get("last_user_input")
    npc_id = state.get("npc_id", "Damon")

    # 1. 短期记忆
    with StageTimer(trace, "short_memory_load"):
        from prompt_construction.prompt.context_formatter import (
            format_short_history,
            format_mid_memory,
            format_long_memory,
            format_today_actions,
        )
        recent_messages = ChatMemory.load(npc_id)
        short_history = format_short_history(recent_messages)

    # 2. 中期记忆
    with StageTimer(trace, "mid_memory_load"):
        from memory.MTMemory import MidTermMemory
        mid_contents = MidTermMemory.load(npc_id)
        mid_memory = format_mid_memory(mid_contents)

    # 3. NPC 配置 + 长期记忆
    with StageTimer(trace, "long_memory_retrieve"):
        from prompt_construction.npc.npc_manager import load_npc_config
        npc_config = load_npc_config(npc_id, state.get("relationship"))

        ctx, memory_raw = global_context_manager.get_context(
            npc_config.get("description", ""),
            npc_config.get("instruction", ""),
            player_input,
            state,
            return_raw=True,
        )
        long_memory = format_long_memory(ctx)

    trace.set_memory(ctx, memory_raw)

    # 4. Few-shot 示例（LanceDB 检索 + 分层 fallback）
    with StageTimer(trace, "few_shot_retrieve"):
        from prompt_construction.retrieval.few_shot_provider import get_few_shot_examples
        few_shot_result = get_few_shot_examples(npc_id, state, npc_config)
        examples_text = few_shot_result.text

    trace.set_few_shot(few_shot_result, few_shot_result.selected_records)

    # Debug log (only source/count, NOT full prompt)
    print(
        f"[few-shot] npc={npc_id} source={few_shot_result.source} "
        f"count={few_shot_result.count} "
        f"reason={few_shot_result.debug.get('fallback_reason', 'none')}"
    )

    # 5. 格式化今日行动
    today_actions_text = format_today_actions(state.get("today_actions", []))

    # 6. 构建 prompt
    with StageTimer(trace, "prompt_build"):
        from prompt_construction.prompt.prompt_builder import build_prompt, PromptContext
        prompt_ctx = PromptContext(
            npc_name=npc_id,
            character_type=npc_config.get("character_type", "native"),
            persona_core=npc_config.get("persona_core", ""),
            persona_background=npc_config.get("persona_background"),
            persona_growth=npc_config.get("persona_growth"),
            speech_style=npc_config.get("speech_style"),
            relationship_desc=npc_config.get("description", ""),
            interaction_style=npc_config.get("instruction", ""),
            persona_memory=long_memory["persona_text"],
            long_term_impression=long_memory["summary_text"],
            relevant_past_events=long_memory["event_text"],
            short_history=short_history,
            today_conversations=mid_memory,
            weather=state.get("weather", "晴朗"),
            game_time=state.get("game_time", "未知"),
            location=state.get("location", "未知"),
            player_info=state.get("player_info", "healthy"),
            today_actions=today_actions_text,
            attitude=state.get("attitude", "中立"),
            dialogue_examples=examples_text,
            gift_rules=npc_config.get("gift", ""),
            player_input=player_input or "",
            mood_rules=npc_config.get("mood_rules"),
            dialogue_constraints=npc_config.get("dialogue_constraints"),
            do_list=npc_config.get("do", []),
            dont_list=npc_config.get("dont", []),
        )
        prompt = build_prompt(prompt_ctx)

    # Record prompt metadata for trace
    memory_item_count = sum(
        1 for layer_items in (memory_raw or {}).values() for _ in layer_items
    )
    has_static_fallback = bool(
        npc_config.get("static_examples", "")
        and npc_config.get("static_examples", "") != "No dialogue examples."
    )
    trace.set_prompt_info(
        prompt_text=prompt,
        few_shot_count=few_shot_result.count,
        memory_count=memory_item_count,
        has_static_fallback=has_static_fallback,
    )

    # --- 阶段 3: 调用与后处理 ---
    llm_error = None
    with StageTimer(trace, "llm_call"):
        try:
            res = llm.invoke(prompt)

            if isinstance(res.content, list):
                res_content = " ".join(
                    str(item['text']) if isinstance(item, dict) and 'text' in item else ''
                    for item in res.content
                )
            else:
                if isinstance(res.content, dict) and 'text' in res.content:
                    res_content = str(res.content['text'])
                else:
                    res_content = str(res.content)

            npc_reply = res_content.strip()
            npc_reply_for_memory = npc_reply

            if "[GIVE:" in npc_reply:
                from gameinfo.give_present import process_category_gifts
                stdout_version, memory_version, _ = process_category_gifts(npc_reply)
                npc_reply = stdout_version
                npc_reply_for_memory = memory_version

        except Exception as e:
            print(f"--- [Error] handle_normal 逻辑崩溃: {e} ---")
            npc_reply = "抱歉，刚才走神了，你刚才说什么？"
            npc_reply_for_memory = npc_reply
            llm_error = str(e)

    # Record LLM output in trace
    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        pass
    trace.set_llm_output(
        raw_output=npc_reply,
        final_output=npc_reply,
        provider=cfg.get("Provider", ""),
        model=cfg.get("ModelName", ""),
        temperature=cfg.get("Temperature", 0),
        error=llm_error,
    )

    with StageTimer(trace, "postprocess"):
        ChatMemory.save(npc_id, "player", player_input)
        ChatMemory.save(npc_id, "npc", npc_reply_for_memory)

    state["npc_reply"] = npc_reply
    return state

def handle_end(state: Dict):
    """Handle end of dialogue: summarize memories, spread gossip, and clean up."""
    from memory.MTMemory import summarize_to_mid_term
    from prompt_construction.npc.gossip import spread_gossip

    npc_id = state.get("npc_id", "Damon")
    summarized_memories = None

    try:
        summarized_memories = summarize_to_mid_term(state)
        if summarized_memories:
            spread_gossip(npc_id, summarized_memories)
    except Exception as e:
        print(f"⚠️ Warning: Memory summarization failed for {npc_id}: {e}")

    ChatMemory.clear(npc_id)
    state["npc_reply"] = None
    state["last_user_input"] = None
    return state

def handle_cancel(state: DialogueState):
    npc_id = state.get("npc_id", "Damon")
    ChatMemory.clear(npc_id)
    state["npc_reply"] = None
    state["last_user_input"] = None
    return state

def handle_end_of_day(state: DialogueState) -> Dict:
    """END_DAY 长期记忆结转：中期记忆 → 长期事件 + 主观偏好 belief + 关系 impression + 轻量 decay"""
    npc_id = state.get("npc_id", "Damon")
    current_day = state.get("time_num", 1)

    from memory.MTMemory import MidTermMemory
    from memory.end_day_pipeline import (
        filter_episodic_events,
        save_episodic_events,
        update_daily_preference_beliefs,
        update_daily_relationship_impression,
        apply_daily_memory_decay,
    )

    # 1. 读取中期记忆
    mid_memories = MidTermMemory.load(npc_id)

    if not mid_memories:
        print("今日无值得回顾的记忆。")
        return state

    # 2. 从中期记忆中筛选 / 升级出候选长期记忆
    candidates = MidTermMemory.upgrade(mid_memories, current_day)

    # 3. 筛选值得保存的 episodic_event
    event_memories = filter_episodic_events(candidates)

    # 4. 写入 episodic_event
    saved_events = save_episodic_events(
        event_memories=event_memories,
        state=state,
        store=store,
    )

    # 5. 根据今日事件更新 preference_belief
    update_daily_preference_beliefs(
        saved_events=saved_events,
        state=state,
        store=store,
    )

    # 6. 根据今日事件更新 relationship_impression
    update_daily_relationship_impression(
        saved_events=saved_events,
        state=state,
        store=store,
    )

    # 7. 对旧 episodic_event 做轻量 decay
    apply_daily_memory_decay(
        npc_id=npc_id,
        current_day=current_day,
        store=store,
    )

    # 8. 清空中期记忆
    MidTermMemory.clear(npc_id)

    print(
        f"今日长期记忆结转完成："
        f"候选 {len(candidates)} 条，"
        f"保存 episodic_event {len(saved_events)} 条。"
    )
    return state

def handle_weekly_reflection(state: DialogueState) -> Dict:
    """WEEKLY_REFLECTION：生成 narrative_arc + 计算 relationship_delta + 标记 consolidated。"""
    current_day = state.get("time_num", 1)
    start_time = current_day - 7
    end_time = current_day

    print(f"[System] 开始周反思 (Day {start_time}-{end_time})...")

    # 1. 找出这一周有戏份的 NPC
    active_npcs = store.get_active_npc_ids(start_time, end_time)
    print(f"本周活跃 NPC: {active_npcs}")

    batch_results = {}

    # 2. 遍历处理
    for npc_id in active_npcs:
        try:
            from memory.consolidation import WeeklyReflector
            result = WeeklyReflector.run_weekly_reflection(
                npc_id=npc_id,
                current_time=current_day,
                store=store,
                llm=llm,
            )

            batch_results[npc_id] = {
                "narrative_arc_id": result.get("narrative_arc_id"),
                "relationship_delta": result.get("relationship_delta", {}),
            }
        except Exception as e:
            print(f"[Error] 处理 {npc_id} 周反思时出错: {e}")

    print(f"周反思完成: {list(batch_results.keys())}")

    # 3. 返回批量数据
    return {
        "npc_reply": None,
        "command": "WEEKLY_REFLECTION",
        "batch_data": batch_results,
    }

def handle_season_end(state: DialogueState) -> Dict:
    """SEASON_END 季节结束重型 consolidation：归档/删除/合并/重建。"""
    current_day = state.get("time_num", 1)

    print(f"[System] 开始季节结束 consolidation (Day {current_day})...")

    from memory.season_end import SeasonConsolidator

    # 1. 获取全部有记忆的 NPC
    all_npc_ids = SeasonConsolidator.get_all_npc_ids(store)
    print(f"本季需 consolidation 的 NPC: {all_npc_ids}")

    batch_results = {}

    # 2. 逐 NPC 执行
    for npc_id in all_npc_ids:
        try:
            result = SeasonConsolidator.run_season_consolidation(
                npc_id=npc_id,
                current_day=current_day,
                store=store,
                llm=llm,
            )
            batch_results[npc_id] = result
        except Exception as e:
            print(f"[Error] 处理 {npc_id} 季节 consolidation 时出错: {e}")

    print(f"季节结束 consolidation 完成: {list(batch_results.keys())}")

    return {
        "npc_reply": None,
        "command": "SEASON_END",
        "batch_data": batch_results,
    }

def handle_post_process(state: Dict):
    """
    后处理节点：负责记忆回写、状态清理等
    """
    print("🔄 [System] Running Memory WriteBack...")

    all_candidates = state.get("retrieved_memories", [])
    

    used_memories = state.get("final_context_memories", [])
    
    now_time = state.get("time_num") 

    from memory.LTMemory import MemoryWriteBack
    writer = MemoryWriteBack() 
    
    writer.update(all_candidates, used_memories, now_time)

    return state

# ===== 增强版 Graph =====
class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {} 
        self.entry = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, from_node, to_node):
        self.edges[from_node] = to_node

    def set_entry_point(self, name):
        self.entry = name

    def run(self, state):
        import time
        current = self.entry
        loop_count = 0
        
        while current:
            node_func = self.nodes.get(current)
            if not node_func: break
            
            start_time = time.time()
            print(f"  >>> 正在运行节点: {current} ...", end="", flush=True)
            state = node_func(state)
            print(f" 完成! 耗时: {time.time() - start_time:.2f}s", flush=True)
            
            next_destination = self.edges.get(current)
            
            if callable(next_destination):
                current = next_destination(state)
            else:
                current = next_destination
            
            loop_count += 1
            if loop_count > 20: break
            
        return state
    
def route_by_command(state):
    cmd = state.get("command", "NORMAL")
    
    if cmd == "NORMAL":
        return "handle_normal"
    elif cmd == "END_DIALOGUE":
        return "handle_end"
    elif cmd == "CLEAR_DIALOGUE":
        return "handle_cancel"
    elif cmd == "END_DAY":
        return "handle_end_of_day"
    elif cmd == "WEEKLY_REFLECTION":
        return "handle_weekly_reflection"
    elif cmd == "SEASON_END":
        return "handle_season_end"
    else:
        return None 
    
# ===== Build Graph =====
graph = StateGraph()
graph.add_node("handle_normal", handle_normal)
graph.add_node("handle_end", handle_end)
graph.add_node("handle_cancel", handle_cancel)
graph.add_node("handle_end_of_day", handle_end_of_day)
graph.add_node("handle_season_end", handle_season_end)

# 增加一个“分发器”节点
def dispatcher(state):
    return state # 负责触发下一步的路由

graph.add_node("start", dispatcher)
graph.set_entry_point("start")

graph.add_edge("start", route_by_command)

# NORMAL 分支：先 ensure_persona -> 再 handle_normal -> 结束
graph.add_edge("handle_normal", None)

# 其他分支：直接运行后结束
graph.add_edge("handle_end", None)
graph.add_edge("handle_cancel", None)
graph.add_edge("handle_end_of_day", None)
graph.add_edge("handle_season_end", None)
