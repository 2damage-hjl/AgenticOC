import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# 1. 引入你在 graph.py 里写好的 graph 对象
# 注意：这里 import graph 时，graph.py 里的全局初始化代码(加载模型等)会自动执行
from graph import graph 

# 创建 FastAPI 实例
app = FastAPI(title="Stardew Valley AI Server")

# 2. 定义请求体 (Request Body)
# 这里必须涵盖 C# 端可能传过来的所有数据
class GameStateInput(BaseModel):
    # --- 控制字段 ---
    command: str = Field(default="NORMAL", description="指令类型: NORMAL, END_DIALOGUE, etc.")
    npc_id: str = Field(default="Damon", description="正在对话的NPC名字")
    
    # --- 玩家输入 ---
    player_input: Optional[str] = Field(default=None, description="玩家说的话")
    
    # --- 游戏环境上下文 (必须与 graph.py 中的 DialogueState 对应) ---
    location: str = "Town"
    relationship: str = "stranger"
    attitude: str = "neutral"
    weather: str = "Sun"
    season: str = "spring"
    year: int =1
    dayOfMonth: int = 1
    
    # --- 玩家状态 ---
    player_info: str = "healthy"
    luckystatus: str = "Neutral"
    today_actions: List[str] = [] # 玩家今天干了啥

    # --- 预留字段 (防止未来加参数报错) ---
    extra: Dict[str, Any] = {}

# 3. 定义 API 接口
@app.get("/")
def health_check():
    """用于 C# 检查服务器是否活着"""
    return {"status": "ok", "message": "AI is ready."}

@app.post("/chat")
def chat_endpoint(data: GameStateInput):
    """
    核心对话接口
    接收 C# 的 JSON -> 转成 Python Dict -> 喂给 Graph -> 返回结果
    """
    print(f"📥 收到请求: [{data.command}] Player: {data.player_input}")
    
    try:
        # A. 将 Pydantic 对象转为字典 (对应 DialogueState)
        state_input = data.model_dump()
        
        # 处理一下字段名不匹配的情况 (如果需要)
        # 例如: Graph 里用的是 last_user_input，但 C# 传的是 player_input
        state_input["last_user_input"] = data.player_input
        
        from gameinfo.read_snapshot import GameTime
        gt = GameTime(
            year=data.year,
            season=data.season,
            day=data.dayOfMonth
        )
        state_input["game_time"] = gt.to_string()
        state_input["time_num"] = gt.to_days()
        
        # B. 运行 Graph
        # 注意：这里调用的是你 graph.py 里定义的 graph.run(state)
        final_state = graph.run(state_input)
        
        # C. 构造返回给 C# 的数据
        # 我们只需要返回 NPC 的回复和可能的指令，不需要把整个 state 传回去
        response = {
            "npc_reply": final_state.get("npc_reply", "..."),
            "command": final_state.get("command", "NORMAL"),
            "error": final_state.get("error", None)
        }
        
        print(f"📤 返回回复: {response['npc_reply'][:30]}...")
        return response

    except Exception as e:
        import traceback
        traceback.print_exc() # 在控制台打印详细报错
        print(f"❌ Server Error: {e}")
        # 返回 500 错误给 C#
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动服务器
    # host="0.0.0.0" 允许局域网访问，"127.0.0.1" 仅限本机
    print("🚀 正在启动 Damon AI 服务器...")
    uvicorn.run(app, host="127.0.0.1", port=8000)