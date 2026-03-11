import json
import os
import sys


# =========================
#  config 加载
# =========================

def get_base_dir():
    # 兼容 PyInstaller
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    base_dir = get_base_dir()
    config_path = os.path.join(base_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json 不存在")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
#  LLM Factory
# =========================

OPENAI_COMPATIBLE = {
    "openai",
    "openrouter",
    "deepseek",
    "qwen"
}

_cached_llm = None

def create_llm():
    global _cached_llm

    if _cached_llm is not None:
        return _cached_llm
    
    cfg = load_config()

    provider = cfg["Provider"].lower()
    model = cfg["ModelName"]
    api_key = cfg["ApiKey"]
    base_url = cfg.get("ServerAddress")
    temperature = cfg.get("Temperature", 0.7)

    print(f"--- [首次初始化] 正在加载 {provider} 引擎和重型 SDK... ---",flush=True)

    # ---------- OpenAI-compatible ----------
    if provider in {"openai", "openrouter", "deepseek", "qwen"}:

        from langchain_openai import ChatOpenAI
        
        kwargs = {
            "model": model,
            "temperature": temperature,
            "openai_api_key": api_key,
        }
        if base_url:
            kwargs["openai_api_base"] = base_url
            
        _cached_llm = ChatOpenAI(**kwargs)

    # ---------- Gemini ----------
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        _cached_llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    else:
        raise ValueError(f"未知 Provider: {provider}")

    return _cached_llm
