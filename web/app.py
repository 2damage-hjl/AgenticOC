import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ====== LLM 调用 ======
def call_llm(api_key, model, prompt, base_url=None):
    """
    通用 LLM 调用函数（兼容 OpenAI / DeepSeek / Qwen）
    """

    if base_url is None:
        base_url = "https://api.openai.com/v1"

    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert NPC persona designer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    # 👉 debug
    print("STATUS:", response.status_code)
    print("RAW RESPONSE:", response.text)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")

    result = response.json()

    if "choices" not in result:
        raise Exception(f"Invalid response: {result}")

    content = result["choices"][0]["message"]["content"]
    return content

def call_gemini(api_key, model, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    print("GEMINI:", response.text)

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()

    return result["candidates"][0]["content"]["parts"][0]["text"]

def smart_call_llm(api_key, model, prompt, base_url=None):
    if "gemini" in model.lower():
        return call_gemini(api_key, model, prompt)
    else:
        return call_llm(api_key, model, prompt, base_url)

# ====== 生成 Python 文件 ======
def generate_python_code(npc_name, data):
    lines = []

    lines.append("import uuid")
    lines.append("from typing import List, Dict\n")

    func_name = f"build_{npc_name.lower()}_persona_seed"

    lines.append(f"def {func_name}() -> List[Dict]:")
    lines.append(f"    npc_id = \"{npc_name}\"\n")

    def write_block(name, items):
        lines.append(f"    {name} = [")
        for item in items:
            content = item["content"].replace('"', '\\"')
            importance = item["importance"]
            lines.append(f"        (\"{content}\", {importance}),")
        lines.append("    ]\n")

    write_block("core_traits", data["traits"])
    write_block("facts", data["facts"])
    write_block("style_samples", data["style_samples"])

    lines.append("    all_texts = core_traits + facts + style_samples\n")

    lines.append("""
    return [
        {
            "memory_id": str(uuid.uuid4()),
            "npc_id": npc_id,
            "content": text,
            "time": "static",
            "location": "persona_seed",
            "importance": importance,
            "memory_type": "persona_seed"
        }
        for text, importance in all_texts
    ]
""")

    return "\n".join(lines)


@app.route("/export", methods=["POST"])
def export():
    data = request.json

    code = generate_python_code(data["npc_name"], data["persona"])

    filename = f"{data['npc_name']}_persona_seed.py"
    path = os.path.join(data["mod_path"], filename)

    try:
        os.makedirs(data["mod_path"], exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        return jsonify({"success": True, "path": path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)