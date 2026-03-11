import json
import os
from typing import Dict, Any

class GameTime:   
    def __init__(self, year=1, season="spring", day=1):
        self.year = year
        self.season = season
        self.day_of_month = day
    def to_string(self) -> str:
        return f"Year {self.year}, {self.season}, Day {self.day_of_month}"
    
    def to_days(self)->int:
        seasons_order = {"spring":0,"summer":1,"fall":2,"winter":3}
        return self.year*120 +seasons_order[self.season]*30 + self.day_of_month

_cached_state = None

def get_snapshot_path():
    """
    智能寻找 snapshot.json：先找当前目录，再找上一级目录。
    """
    # 1. 尝试脚本所在目录
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "snapshot.json")
    if os.path.exists(path):
        return path
    
    # 2. 尝试上一级目录 (根目录)
    root = os.path.dirname(base)
    path = os.path.join(root, "snapshot.json")
    if os.path.exists(path):
        return path
    
    # 3. 实在找不到，回退到当前工作目录
    return os.path.join(os.getcwd(), "snapshot.json")

def _perform_load_io():
    """私有方法：真正执行繁重的磁盘读取逻辑"""
    path = get_snapshot_path()
    # 1. 尝试脚本所在目录

    if not os.path.exists(path):
        # 如果是循环对话模式，找不到快照时可以给一个默认值，防止程序直接崩溃
        print("⚠️ 警告: snapshot.json 不存在，使用默认环境参数")
        raw = {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    # 时间处理
    gt = GameTime(
        year=raw.get("year", 1),
        season=raw.get("season", "spring"),
        day=raw.get("dayOfMonth", 1)
    )

    state = {
        "npc_id": raw.get("npcName", "Unknown"),
        "game_time": gt.to_string(),
        "time_num": gt.to_days(),
        "location": raw.get("location", "Unknown"),
        "altitude": raw.get("altitude", "neutral"),
        "relationship": raw.get("relationship", "neutral"),
        "weather": raw.get("weather", "sunny"),
        "player_info": raw.get("playerinfo", "healthy"),
        "today_actions": raw.get("todayActions", []),
        "luckystatus": raw.get("luckStatus", "normal"),
        "command": raw.get("command", "NORMAL"),
        "last_user_input": None,
        "npc_reply": None,
        "error": None
    }
    
    print("环境快照已初始化")
    return state 

#=========接口=========
def get_state(force_refresh: bool = False) -> Dict[str, Any]:
    global _cached_state
    
    # 逻辑：如果没有缓存，或者被外部强制刷新，才进行文件 IO
    #TODO:可能后续需要修改
    if _cached_state is None or force_refresh:
        _cached_state = _perform_load_io()
        print("--- [IO] 已重新读取环境快照 ---", flush=True)
        
    return _cached_state

def reset_state():
    """清理状态，下次调用 get_state 会重新触发磁盘读取"""
    global _cached_state
    _cached_state = None
    print("--- [State] 状态已重置 ---", flush=True)