from langgraph.graph import StateGraph, END
import os
from typing import List, TypedDict, Literal, Optional, Dict
from memory.STMemory import ChatMemory
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
print(">>> 模型加载完毕！(后续对话将不再卡顿)")
retriever = MemoryRetriever(store)# 绑定检索器
ranker = MemoryRanker()           # 绑定排序器
global_context_manager = ContextManager(retriever, ranker) 
print("系统初始化完成！")
from llm import create_llm
llm = create_llm()

INIT_FLAG = ".persona_init_done"
from memory.persona_seed import initial_persona_seed, build_damon_persona_seed
if not os.path.exists(INIT_FLAG):
    print("--- 正在进行首次人设初始化 (写入向量库) ---")
    try:
        initial_persona_seed(
            npc_id="Damon", 
            memory_store=store,
            build_func=build_damon_persona_seed
        )
        
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
    player_input = state.get("last_user_input")
    npc_id = state.get("npc_id", "Damon")

    # 短期记忆
    recent_messages = ChatMemory.load(npc_id)
    formatted_history = "\n".join([
        f"{'玩家' if m.get('role')=='player' else 'npc'}: {m.get('content', '')}" 
        for m in recent_messages
    ])
    
    #中期记忆
    from memory.MTMemory import MidTermMemory
    mid_contents = MidTermMemory.load(npc_id)
    print(f"成功导入{npc_id}中期记忆")

    formatted_today_mems = []
    for mem in mid_contents:
        m_str = f"[地点:{mem['location']}] {mem['content']}"
        formatted_today_mems.append(m_str)
        print(f"中期记忆: {m_str}")

    today_conversations = "\n".join([f"- {m}" for m in formatted_today_mems]) if formatted_today_mems else "暂无"

    #长期记忆
    from npc.npc_manager import load_relationship_config
    disc,instr,gift,dialogua_instance = load_relationship_config(npc_id,state.get("relationship"))
    ctx = global_context_manager.get_context(disc,instr,player_input, state)
   
    # --- 阶段 3: 调用与后处理 ---
    try:
        if npc_id == "Damon":
            from prompt.prompt_damon import get_prompt
            prompt = get_prompt(npc_id, state, disc, instr,gift, dialogua_instance, formatted_history, today_conversations, ctx, player_input)
        else:
            from prompt.prompt_others import get_prompt
            prompt = get_prompt(npc_id, state, disc, instr, gift, dialogua_instance, formatted_history, today_conversations, ctx, player_input)
        res = llm.invoke(prompt)

        # 1. 获取原始内容
        #raw_content = res.content
        #print(f"ai原始输出内容: {raw_content}")
        
        if isinstance(res.content, list):
            # 提取每个字典中的 'text' 字段，并确保它是字符串类型
            res_content = " ".join(
                str(item['text']) if isinstance(item, dict) and 'text' in item else ''
                for item in res.content
            )
        else:
            # 如果不是列表，而是字典，提取 'text' 字段
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
        print(f"--- [Error] Damon 逻辑崩溃: {e} ---")
        npc_reply = "抱歉，刚才走神了，你刚才说什么？"
        npc_reply_for_memory = npc_reply

    ChatMemory.save(npc_id, "player", player_input)
    ChatMemory.save(npc_id, "npc", npc_reply_for_memory)

    state["npc_reply"] = npc_reply
    return state

def handle_end(state: Dict):
    """Handle end of dialogue: summarize memories, spread gossip, and clean up."""
    from memory.MTMemory import summarize_to_mid_term
    from npc.gossip import spread_gossip

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

def handle_end_of_day(state: DialogueState) -> Dict: # 移除 memory_store 参数
    npc_id = state.get("npc_id", "Damon")
    
    from memory.MTMemory import MidTermMemory
    mid_memories = MidTermMemory.load(npc_id) 
    
    if not mid_memories:
        print("今日无值得回顾的记忆。")
        return state # 记得返回 state

    memories_to_save = MidTermMemory.upgrade(mid_memories, state["time_num"])

    if memories_to_save:
        print(f"正在向量化并存储 {len(memories_to_save)} 条长期记忆...")
        # ✅ 不需要 import MemoryStore，直接用全局变量 store
        for mem in memories_to_save:
            store.add(  # ✅ 使用全局实例 store
                layer="event", 
                content=mem["content"], 
                metadata={
                    "npc_id": mem["npc_id"],
                    "time": mem["time"],
                    "location": mem["location"],
                    "importance": mem["importance"],
                    "memory_type": "event"
                },
                doc_id=mem["memory_id"]
            )
    
    MidTermMemory.clear(npc_id)
    print("今日记忆结转完成。")
    return state

def handle_summary_all(state: DialogueState) -> Dict:
    current_day = state.get("time_num", 1)
    # 过去 7 天
    start_time = current_day - 7
    end_time = current_day

    print(f"📅 [System] 开始周结 (Day {start_time}-{end_time})...")

    # 1. 找出这一周有戏份的 NPC
    active_npcs = store.get_active_npc_ids(start_time, end_time)
    print(f"📋 本周活跃 NPC: {active_npcs}")

    batch_updates = {} # 用于存 {"Damon": 20, "Abigail": 10}

    # 2. 遍历处理
    for npc_id in active_npcs:
        try:
            from memory.consolidation import WeeklyReflector
            # 复用之前的 WeeklyReflector 逻辑
            delta = WeeklyReflector.run_weekly_reflection(
                npc_id=npc_id,
                current_time=current_day,
                store=store,
                llm=llm
            )
            
            # 只有分数有变化才记录
            if delta != 0:
                batch_updates[npc_id] = delta
                
        except Exception as e:
            print(f"❌ 处理 {npc_id} 周结时出错: {e}")

    print(f"✅ 批量周结完成: {batch_updates}")

    # 3. 返回批量数据
    return {
        "npc_reply": "...",
        "command": "BATCH_UPDATE_FRIENDSHIP", # 这是一个新指令
        "batch_data": batch_updates # 直接传字典
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
    elif cmd == "WEEKLY_SUMMARY":
        return "handle_summary_all"
    else:
        return None 
    
# ===== Build Graph =====
graph = StateGraph()
graph.add_node("handle_normal", handle_normal)
graph.add_node("handle_end", handle_end)
graph.add_node("handle_cancel", handle_cancel)
graph.add_node("handle_end_of_day", handle_end_of_day)

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
