"""OC Builder Web Routes — Flask blueprint for the OC character builder UI."""

from __future__ import annotations

import json
import os
import sys
from flask import Blueprint, render_template, request, jsonify

oc_bp = Blueprint("oc_builder", __name__)


def _get_data_dir() -> str:
    """PyInstaller-compatible data directory."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _oc_upload_dir() -> str:
    """Directory for pending OC uploads."""
    path = os.path.join(_get_data_dir(), "oc_uploads")
    os.makedirs(path, exist_ok=True)
    return path


def _npc_dir() -> str:
    """Directory where NPC JSON configs are stored."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompt_construction", "npc"
    )


def _relation_map_path() -> str:
    """Path to relation_map.py."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompt_construction", "npc", "relation_map.py"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@oc_bp.route("/oc")
def oc_builder_page():
    """Serve the OC builder web page."""
    return render_template("oc_builder.html")


@oc_bp.route("/oc/list", methods=["GET"])
def list_ocs():
    """List all OCs that have been installed (have a JSON config in npc dir)."""
    npc_dir = _npc_dir()
    ocs = []
    if os.path.isdir(npc_dir):
        for fname in os.listdir(npc_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(npc_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("character_type") == "original":
                        ocs.append(data.get("npc_id", fname.replace(".json", "")))
                except Exception:
                    pass
    return jsonify({"ocs": ocs})


@oc_bp.route("/oc/build", methods=["POST"])
def build_oc():
    """
    Main endpoint: receive OC form data, generate NPC config + persona seed,
    and write to oc_uploads/ for auto-import on next server start.
    """
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    oc_name = data.get("oc_name", "").strip()
    if not oc_name:
        return jsonify({"success": False, "error": "OC name is required"}), 400

    # Validate name (no special chars, no path traversal)
    if not oc_name.replace("_", "").replace(" ", "").isalnum():
        return jsonify({"success": False, "error": "OC name must be alphanumeric"}), 400

    try:
        upload_dir = _oc_upload_dir()
        oc_dir = os.path.join(upload_dir, oc_name)
        os.makedirs(oc_dir, exist_ok=True)

        # 1. Generate NPC JSON config
        npc_config = _build_npc_config(data)
        npc_path = os.path.join(oc_dir, f"{oc_name}.json")
        with open(npc_path, "w", encoding="utf-8") as f:
            json.dump(npc_config, f, ensure_ascii=False, indent=2)

        # 2. Generate persona seed data
        persona_seed = _build_persona_seed(data)
        seed_path = os.path.join(oc_dir, "persona_seed.json")
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(persona_seed, f, ensure_ascii=False, indent=2)

        # 3. Generate relation map additions
        relations = data.get("relations", {})
        if relations:
            rel_path = os.path.join(oc_dir, "relations.json")
            with open(rel_path, "w", encoding="utf-8") as f:
                json.dump(relations, f, ensure_ascii=False, indent=2)

        # 4. Save dialogue examples if provided
        dialogue_examples = data.get("dialogue_examples", [])
        if dialogue_examples:
            dlg_path = os.path.join(oc_dir, "dialogue_examples.json")
            with open(dlg_path, "w", encoding="utf-8") as f:
                json.dump(dialogue_examples, f, ensure_ascii=False, indent=2)

        # 5. Save dialogue.json if uploaded
        dialogue_json = data.get("dialogue_json")
        if dialogue_json:
            dlg_json_path = os.path.join(oc_dir, "dialogue.json")
            with open(dlg_json_path, "w", encoding="utf-8") as f:
                json.dump(dialogue_json, f, ensure_ascii=False, indent=2)

        # 6. Auto-install immediately
        _install_oc(oc_name, oc_dir)

        msg = (
            f"配置文件已写入 oc_uploads/{oc_name}/ 并自动安装。"
            f"重启游戏后即可与 {oc_name} 对话。"
        )
        return jsonify({"success": True, "message": msg})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# NPC config builder
# ---------------------------------------------------------------------------

def _build_npc_config(data: dict) -> dict:
    """Build the NPC JSON config from form data (same format as Damon.json)."""
    oc_name = data["oc_name"]

    config = {
        "npc_id": oc_name,
        "character_type": data.get("character_type", "original"),
        "persona_core": data.get("persona_core", ""),
        "persona_background": data.get("persona_background") or None,
        "persona_growth": data.get("persona_growth") or None,
        "speech_style": data.get("speech_style") or None,
        "mood_rules": data.get("mood_rules") or None,
        "dialogue_constraints": data.get("dialogue_constraints") or None,
        "do": data.get("do_list", []),
        "dont": data.get("dont_list", []),
        "relationship_map": data.get("relationship_map", {}),
    }

    # Fallback: if no relationship_map stages provided, create minimal ones
    if not config["relationship_map"]:
        config["relationship_map"] = {
            "stranger": {
                "description": f"完全的陌生人。{oc_name} 保持警惕和疏离。",
                "instruction": "保持礼貌但疏离。句子简短。",
                "gift": "不送礼",
                "examples": [],
            },
            "acquaintance": {
                "description": f"点头之交。{oc_name} 开始记得对方。",
                "instruction": "可以进行基本寒暄，但仍保持距离。",
                "gift": "不送礼",
                "examples": [],
            },
            "friend": {
                "description": f"朋友。{oc_name} 愿意分享一些自己的事。",
                "instruction": "语气放松，愿意开启话题。",
                "gift": "不送礼",
                "examples": [],
            },
            "close friend": {
                "description": f"密友。{oc_name} 在对方面前展露真实的一面。",
                "instruction": "流露真情实感，会主动关心对方。",
                "gift": "偶尔会送小礼物",
                "examples": [],
            },
            "best friend": {
                "description": f"挚友。无条件的信任。",
                "instruction": "完全敞开，全力支持。",
                "gift": "经常会送礼物",
                "examples": [],
            },
        }

    return config


# ---------------------------------------------------------------------------
# Persona seed builder
# ---------------------------------------------------------------------------

def _build_persona_seed(data: dict) -> list:
    """Build persona seed list from form data (same format as build_damon_persona_seed)."""
    import uuid

    oc_name = data["oc_name"]
    memories = []

    # Core traits
    for trait in data.get("traits", []):
        memories.append({
            "memory_id": str(uuid.uuid4()),
            "npc_id": oc_name,
            "content": trait["content"],
            "time": "static",
            "location": "persona_seed",
            "importance": trait.get("importance", 0.75),
            "memory_type": "persona_seed",
            "status": "active",
            "last_access": 0,
            "source": "oc_builder",
        })

    # Facts
    for fact in data.get("facts", []):
        memories.append({
            "memory_id": str(uuid.uuid4()),
            "npc_id": oc_name,
            "content": fact["content"],
            "time": "static",
            "location": "persona_seed",
            "importance": fact.get("importance", 0.70),
            "memory_type": "persona_seed",
            "status": "active",
            "last_access": 0,
            "source": "oc_builder",
        })

    return memories


# ---------------------------------------------------------------------------
# Auto-install: copy files to correct locations
# ---------------------------------------------------------------------------

def _install_oc(oc_name: str, oc_dir: str):
    """Install an OC from oc_uploads/{oc_name}/ to the active system paths."""
    npc_dir = _npc_dir()

    # 1. Copy NPC JSON config
    src_npc = os.path.join(oc_dir, f"{oc_name}.json")
    dst_npc = os.path.join(npc_dir, f"{oc_name}.json")
    if os.path.exists(src_npc):
        with open(src_npc, "r", encoding="utf-8") as f:
            npc_data = json.load(f)
        with open(dst_npc, "w", encoding="utf-8") as f:
            json.dump(npc_data, f, ensure_ascii=False, indent=2)
        print(f"[OC Builder] NPC 配置已安装: {dst_npc}")

    # 2. Inject persona seed into ChromaDB
    src_seed = os.path.join(oc_dir, "persona_seed.json")
    if os.path.exists(src_seed):
        with open(src_seed, "r", encoding="utf-8") as f:
            seed_data = json.load(f)
        _import_persona_seed(oc_name, seed_data)

    # 3. Merge relations into relation_map.py
    src_rel = os.path.join(oc_dir, "relations.json")
    if os.path.exists(src_rel):
        with open(src_rel, "r", encoding="utf-8") as f:
            relations = json.load(f)
        _merge_relation_map(oc_name, relations)

    # 4. Mark as installed
    install_flag = os.path.join(oc_dir, ".installed")
    with open(install_flag, "w") as f:
        f.write("done")

    print(f"[OC Builder] OC '{oc_name}' 安装完成！")


def _import_persona_seed(oc_name: str, seed_data: list):
    """Import persona seed memories into ChromaDB."""
    try:
        # Lazy import to avoid circular dependency and heavy model loading
        from memory.embedded import MemoryStore
        store = MemoryStore()

        # Check if already exists
        existing = store.query(
            layer="persona_seed",
            filter={"npc_id": oc_name, "source": "oc_builder"},
            limit=1
        )
        if existing and existing.get("ids"):
            print(f"[OC Builder] {oc_name} 的 persona_seed 已存在，跳过导入")
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
                doc_id=m["memory_id"]
            )
        print(f"[OC Builder] {oc_name}: {len(seed_data)} 条 persona_seed 已导入 ChromaDB")

    except Exception as e:
        print(f"[OC Builder] persona_seed 导入失败 (将在下次启动时重试): {e}")


def _merge_relation_map(oc_name: str, relations: dict):
    """Merge OC's relations into the SOCIAL_GRAPH in relation_map.py."""
    rel_path = _relation_map_path()

    if not os.path.exists(rel_path):
        print(f"[OC Builder] relation_map.py 不存在: {rel_path}")
        return

    with open(rel_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if OC already in SOCIAL_GRAPH
    if f'"{oc_name}"' in content:
        print(f"[OC Builder] {oc_name} 已在 SOCIAL_GRAPH 中，跳过合并")
        return

    # Build the new entry string
    entries = []
    for target, decay in relations.items():
        entries.append(f'        "{target}": {decay}')
    entries_str = ",\n".join(entries)

    new_entry = f'    "{oc_name}": {{\n{entries_str}\n    }}'

    # Insert before the closing brace of SOCIAL_GRAPH
    # Find the last }
    last_brace = content.rfind("}")
    if last_brace == -1:
        print(f"[OC Builder] 无法解析 relation_map.py")
        return

    # Find the second-to-last } (end of last entry)
    # Insert a comma after the last entry and add new entry
    insert_pos = content.rfind("}", 0, last_brace)
    if insert_pos == -1:
        insert_pos = last_brace

    new_content = content[:insert_pos + 1] + ",\n" + new_entry + "\n" + content[last_brace:]

    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[OC Builder] {oc_name} 的关系网已合并到 relation_map.py")


# ---------------------------------------------------------------------------
# Auto-import on server start (called from server.py)
# ---------------------------------------------------------------------------

def auto_import_pending_ocs():
    """Scan oc_uploads/ for uninstalled OCs and install them."""
    upload_dir = _oc_upload_dir()
    if not os.path.isdir(upload_dir):
        return

    for name in os.listdir(upload_dir):
        oc_dir = os.path.join(upload_dir, name)
        if not os.path.isdir(oc_dir):
            continue

        install_flag = os.path.join(oc_dir, ".installed")
        if os.path.exists(install_flag):
            continue

        print(f"[OC Builder] 发现待安装的 OC: {name}")
        try:
            _install_oc(name, oc_dir)
        except Exception as e:
            print(f"[OC Builder] 安装 {name} 失败: {e}")
