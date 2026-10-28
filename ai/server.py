import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import sys
import json
import uuid

# 1. 引入你在 graph.py 里写好的 graph 对象
# 注意：这里 import graph 时，graph.py 里的全局初始化代码(加载模型等)会自动执行
from graph import graph 

# 创建 FastAPI 实例
app = FastAPI(title="Stardew Valley AI Server")

# CORS — 允许 OC Builder 页面访问 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 定义请求体 (Request Body)
# 这里必须涵盖 C# 端可能传过来的所有数据
class GameStateInput(BaseModel):
    # --- 控制字段 ---
    command: str = Field(default="NORMAL", description="指令类型: NORMAL, END_DIALOGUE, etc.")
    npc_id: str = Field(default="", description="正在对话的NPC名字")
    
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

    # --- Few-shot 状态 (scene_type 过滤 / route / flags) ---
    is_birthday: bool = False
    is_festival: bool = False
    is_gifting: bool = False
    route: str = "community_center_completed"  # pre_choice | community_center_completed | joja_active | joja_completed
    game_flags: List[str] = []  # e.g. ["relationship.married_to.Abigail", "world.ginger_island.beach_resort.opened"]

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
    from prompt_construction.utils.dialogue_trace import DialogueTrace

    trace = DialogueTrace()
    t_total_start = time.perf_counter()

    print(f"📥 收到请求: [{data.command}] Player: {data.player_input}")
    
    try:
        t_parse_start = time.perf_counter()

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

        # Inject trace object into state so graph.py can record per-stage data
        state_input["_trace"] = trace

        trace.set_game_state(state_input)
        trace.set_timing("server_parse", (time.perf_counter() - t_parse_start) * 1000)
        
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
        
        trace.set_timing("total", (time.perf_counter() - t_total_start) * 1000)
        trace.flush()

        print(f"📤 返回回复: {response['npc_reply'][:30]}... | trace={trace._data['trace_id']}")
        return response

    except Exception as e:
        trace.set_timing("total", (time.perf_counter() - t_total_start) * 1000)
        trace._data["error"] = str(e)
        trace.flush()

        import traceback
        traceback.print_exc() # 在控制台打印详细报错
        print(f"❌ Server Error: {e}")
        # 返回 500 错误给 C#
        raise HTTPException(status_code=500, detail=str(e))

# ====== OC Builder Web UI ======

def _web_templates_dir() -> str:
    """Find web templates directory (PyInstaller-compatible)."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "web", "templates")


@app.get("/oc", response_class=HTMLResponse)
def oc_builder_page():
    """Serve the OC Builder web UI."""
    template_path = os.path.join(_web_templates_dir(), "oc_builder.html")
    if not os.path.exists(template_path):
        return HTMLResponse("<h1>OC Builder template not found</h1>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ====== OC Builder API Endpoints ======

def _get_data_dir() -> str:
    """PyInstaller-compatible data directory."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _oc_upload_dir() -> str:
    path = os.path.join(_get_data_dir(), "oc_uploads")
    os.makedirs(path, exist_ok=True)
    return path


def _npc_config_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_construction", "npc")


def _relation_map_path() -> str:
    return os.path.join(_npc_config_dir(), "relation_map.py")


class OCBuilddRequest(BaseModel):
    oc_name: str
    character_type: str = "original"
    persona_core: str = ""
    persona_background: Optional[str] = None
    persona_growth: Optional[str] = None
    speech_style: Optional[str] = None
    mood_rules: Optional[str] = None
    dialogue_constraints: Optional[str] = None
    do_list: List[str] = []
    dont_list: List[str] = []
    relations: Dict[str, float] = {}
    relationship_map: Dict[str, Any] = {}
    traits: List[Dict[str, Any]] = []
    facts: List[Dict[str, Any]] = []
    dialogue_examples: List[Dict[str, str]] = []
    dialogue_json: Optional[Any] = None


@app.get("/oc/list")
def list_ocs():
    """List all installed OC characters."""
    npc_dir = _npc_config_dir()
    ocs = []
    if os.path.isdir(npc_dir):
        for fname in os.listdir(npc_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(npc_dir, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("character_type") == "original":
                        ocs.append(data.get("npc_id", fname.replace(".json", "")))
                except Exception:
                    pass
    return {"ocs": ocs}


@app.post("/oc/build")
def build_oc(data: OCBuilddRequest):
    """Build and install an OC character from form data."""
    oc_name = data.oc_name.strip()
    if not oc_name:
        raise HTTPException(status_code=400, detail="OC name is required")
    if not oc_name.replace("_", "").replace(" ", "").isalnum():
        raise HTTPException(status_code=400, detail="OC name must be alphanumeric")

    try:
        upload_dir = _oc_upload_dir()
        oc_dir = os.path.join(upload_dir, oc_name)
        os.makedirs(oc_dir, exist_ok=True)

        # 1. Generate NPC JSON config
        npc_config = {
            "npc_id": oc_name,
            "character_type": data.character_type,
            "persona_core": data.persona_core,
            "persona_background": data.persona_background,
            "persona_growth": data.persona_growth,
            "speech_style": data.speech_style,
            "mood_rules": data.mood_rules,
            "dialogue_constraints": data.dialogue_constraints,
            "do": data.do_list,
            "dont": data.dont_list,
            "relationship_map": data.relationship_map or _default_relationship_map(oc_name),
        }
        with open(os.path.join(oc_dir, f"{oc_name}.json"), "w", encoding="utf-8") as f:
            json.dump(npc_config, f, ensure_ascii=False, indent=2)

        # 2. Generate persona seed
        seed_data = []
        for t in data.traits:
            seed_data.append(_make_seed_entry(oc_name, t["content"], t.get("importance", 0.75)))
        for fact in data.facts:
            seed_data.append(_make_seed_entry(oc_name, fact["content"], fact.get("importance", 0.70)))
        with open(os.path.join(oc_dir, "persona_seed.json"), "w", encoding="utf-8") as f:
            json.dump(seed_data, f, ensure_ascii=False, indent=2)

        # 3. Relations
        if data.relations:
            with open(os.path.join(oc_dir, "relations.json"), "w", encoding="utf-8") as f:
                json.dump(data.relations, f, ensure_ascii=False, indent=2)

        # 4. Dialogue examples
        if data.dialogue_examples:
            with open(os.path.join(oc_dir, "dialogue_examples.json"), "w", encoding="utf-8") as f:
                json.dump(data.dialogue_examples, f, ensure_ascii=False, indent=2)

        # 5. Dialogue JSON
        if data.dialogue_json:
            with open(os.path.join(oc_dir, "dialogue.json"), "w", encoding="utf-8") as f:
                json.dump(data.dialogue_json, f, ensure_ascii=False, indent=2)

        # 6. Auto-install immediately
        _install_oc(oc_name, oc_dir)

        return {"success": True, "message": f"OC「{oc_name}」已创建并安装！重启游戏后即可对话。"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _make_seed_entry(npc_id: str, content: str, importance: float) -> dict:
    return {
        "memory_id": str(uuid.uuid4()),
        "npc_id": npc_id,
        "content": content,
        "time": "static",
        "location": "persona_seed",
        "importance": importance,
        "memory_type": "persona_seed",
        "status": "active",
        "last_access": 0,
        "source": "oc_builder",
    }


def _default_relationship_map(oc_name: str) -> dict:
    stages = ["stranger", "acquaintance", "friend", "close friend", "best friend"]
    result = {}
    for stage in stages:
        result[stage] = {
            "description": f"与{oc_name}处于{stage}关系。",
            "instruction": "",
            "gift": "不送礼",
            "examples": [],
        }
    return result


def _install_oc(oc_name: str, oc_dir: str):
    """Install an OC from oc_uploads/{oc_name}/ to active system paths."""
    npc_dir = _npc_config_dir()

    # 1. Copy NPC JSON
    src_npc = os.path.join(oc_dir, f"{oc_name}.json")
    dst_npc = os.path.join(npc_dir, f"{oc_name}.json")
    if os.path.exists(src_npc):
        with open(src_npc, "r", encoding="utf-8") as f:
            npc_data = json.load(f)
        with open(dst_npc, "w", encoding="utf-8") as f:
            json.dump(npc_data, f, ensure_ascii=False, indent=2)
        print(f"[OC Builder] NPC 配置已安装: {dst_npc}")

    # 2. Import persona seed into ChromaDB
    src_seed = os.path.join(oc_dir, "persona_seed.json")
    if os.path.exists(src_seed):
        with open(src_seed, "r", encoding="utf-8") as f:
            seed_data = json.load(f)
        _import_persona_seed(oc_name, seed_data)

    # 3. Merge relations
    src_rel = os.path.join(oc_dir, "relations.json")
    if os.path.exists(src_rel):
        with open(src_rel, "r", encoding="utf-8") as f:
            relations = json.load(f)
        _merge_relation_map(oc_name, relations)

    # 4. Mark installed
    with open(os.path.join(oc_dir, ".installed"), "w") as f:
        f.write("done")

    print(f"[OC Builder] OC '{oc_name}' 安装完成！")


def _import_persona_seed(oc_name: str, seed_data: list):
    """Import persona seed into ChromaDB."""
    try:
        from memory.embedded import MemoryStore
        store = MemoryStore()

        existing = store.query(
            layer="persona_seed",
            filter={"npc_id": oc_name, "source": "oc_builder"},
            limit=1,
        )
        if existing and existing.get("ids"):
            print(f"[OC Builder] {oc_name} 的 persona_seed 已存在，跳过")
            return

        for m in seed_data:
            store.add(
                "persona_seed",
                m["content"],
                metadata={
                    "npc_id": m["npc_id"],
                    "importance": m["importance"],
                    "time": m["time"],
                    "location": m["location"],
                    "memory_id": m["memory_id"],
                    "memory_type": m["memory_type"],
                    "status": m.get("status", "active"),
                    "last_access": m.get("last_access", 0),
                    "source": m.get("source", "oc_builder"),
                },
                doc_id=m["memory_id"],
            )
        print(f"[OC Builder] {oc_name}: {len(seed_data)} 条 persona_seed 已导入")
    except Exception as e:
        print(f"[OC Builder] persona_seed 导入失败 (将在下次启动时重试): {e}")


def _merge_relation_map(oc_name: str, relations: dict):
    """Merge OC relations into SOCIAL_GRAPH."""
    rel_path = _relation_map_path()
    if not os.path.exists(rel_path):
        return

    with open(rel_path, "r", encoding="utf-8") as f:
        content = f.read()

    if f'"{oc_name}"' in content:
        print(f"[OC Builder] {oc_name} 已在 SOCIAL_GRAPH 中，跳过")
        return

    entries = []
    for target, decay in relations.items():
        entries.append(f'        "{target}": {decay}')
    entries_str = ",\n".join(entries)
    new_entry = f'    "{oc_name}": {{\n{entries_str}\n    }}'

    last_brace = content.rfind("}")
    if last_brace == -1:
        return

    insert_pos = content.rfind("}", 0, last_brace)
    if insert_pos == -1:
        insert_pos = last_brace

    new_content = content[:insert_pos + 1] + ",\n" + new_entry + "\n" + content[last_brace:]
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[OC Builder] {oc_name} 的关系网已合并")


def _auto_import_pending_ocs():
    """Scan oc_uploads/ for uninstalled OCs."""
    upload_dir = _oc_upload_dir()
    if not os.path.isdir(upload_dir):
        return

    for name in os.listdir(upload_dir):
        oc_dir = os.path.join(upload_dir, name)
        if not os.path.isdir(oc_dir):
            continue
        if os.path.exists(os.path.join(oc_dir, ".installed")):
            continue
        print(f"[OC Builder] 发现待安装的 OC: {name}")
        try:
            _install_oc(name, oc_dir)
        except Exception as e:
            print(f"[OC Builder] 安装 {name} 失败: {e}")


if __name__ == "__main__":
    # 启动服务器
    # host="0.0.0.0" 允许局域网访问，"127.0.0.1" 仅限本机
    print("🚀 正在启动 AgenticOC AI 服务器...")

    # Auto-import pending OCs before starting
    _auto_import_pending_ocs()

    uvicorn.run(app, host="127.0.0.1", port=8000)
