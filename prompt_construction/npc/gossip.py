import random
from npc.relation_map import SOCIAL_GRAPH
from memory.MTMemory import MidTermMemory

def spread_gossip(source_npc_id: str, memories: list):
    """
    八卦传播主函数 (只传播最新的一条)
    Args:
        source_npc_id: 消息源头（当前对话结束的 NPC）
        memories: 中期记忆列表
    """
    # 1. 基础检查：没人或者没记忆就直接跳过
    network = SOCIAL_GRAPH.get(source_npc_id)
    if not network or not memories:
        return 

    # ✅ 【关键修改】只取最新的一条记忆（列表的最后一个元素）
    # 假设 summarize_to_mid_term 是按时间顺序 append 的，那 [-1] 就是刚发生的那件事
    if isinstance(memories, dict):
        # 情况A: 如果传入的是字典，说明这就是最新的一条记忆，直接用
        latest_mem = memories
    elif isinstance(memories, list):
        # 情况B: 如果传入的是列表，取最后一条
        if len(memories) == 0: return
        latest_mem = memories[-1]
    else:
        print(f"--- [Gossip Error] memories 类型错误: {type(memories)} ---")
        return
    
    print(f"--- [Gossip] {source_npc_id} 试图传播最新八卦: {latest_mem.get('content')[:15]}... ---")

    original_weight = latest_mem.get("weight", 0.5)
    original_content = latest_mem.get("content", "")
    
    # 如果这条记忆本身权重太低（比如只是打个招呼），就没必要传播了，节省算力
    if original_weight < 0.2:
        return

    # 2. 尝试传给社交圈里的每一个人
    for target_npc, cost in network.items():
        
        # === A. 计算传播后的新权重 ===
        new_weight = original_weight - cost
        
        # 如果扣完是个负数，直接忽略
        if new_weight <= 0:
            continue

        # === B. 判定是否传播成功 (修改版) ===
        should_propagate = False
        
        # 规则 1: 重大八卦必传 (权重 > 0.6)
        if new_weight > 0.6:
            should_propagate = True
            print(f"  🔥 重大八卦必传 -> {target_npc} (权重 {new_weight:.2f} > 0.6)")
            
        # 规则 2: 普通八卦看运气
        else:
            random_threshold = random.uniform(0.3, 1.0)
            if new_weight > random_threshold:
                should_propagate = True
                print(f"  ✅ 传播成功 -> {target_npc} (权重 {new_weight:.2f} > 阈值 {random_threshold:.2f})")
            # else: 
            #    print(f"  ❌ 传播失败 -> {target_npc}")

        # === C. 执行写入 ===
        if should_propagate:
       
            gossip_content = f"{source_npc_id}以第一人称告诉我：{original_content}"
            
            MidTermMemory.add_gossip_entry(
                npc_id=target_npc,
                content=gossip_content,
                new_weight=new_weight, 
                location=latest_mem.get("location", "Unknown"),
                time=latest_mem.get("time", "Unknown")
            )