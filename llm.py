import json
import os
import sys

from dotenv import load_dotenv

# 项目启动时自动加载 .env
load_dotenv()


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
#  API Key 解析
# =========================

# Provider → 环境变量名的映射
_PROVIDER_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def resolve_api_key(provider: str) -> str:
    """优先从环境变量获取 API Key，回退到 config.json 的 ApiKey 字段。"""
    env_key = _PROVIDER_ENV_KEY.get(provider.lower())
    if env_key:
        key = os.environ.get(env_key, "").strip()
        if key:
            return key

    # 回退：从 config.json 读取（向后兼容）
    cfg = load_config()
    key = cfg.get("ApiKey", "").strip()
    if key:
        return key

    raise ValueError(
        f"未找到 {provider} 的 API Key。"
        f"请设置环境变量 {env_key} 或在 config.json 中配置 ApiKey。"
    )


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

def create_llm(model_name: str | None = None, temperature: float | None = None,
               use_cache: bool = True):
    """创建 LLM 实例。

    Args:
        model_name: 覆盖 config.json 的 ModelName，None 则用配置文件。
        temperature: 覆盖 config.json 的 Temperature，None 则用配置文件。
        use_cache: True 则复用全局单例（仅当无覆盖参数时生效）。
    """
    global _cached_llm

    # 有覆盖参数时，不走缓存，每次新建实例
    if use_cache and model_name is None and temperature is None and _cached_llm is not None:
        return _cached_llm

    cfg = load_config()

    provider = cfg["Provider"].lower()
    model = model_name if model_name else cfg["ModelName"]
    api_key = resolve_api_key(provider)
    base_url = cfg.get("ServerAddress")
    temp = temperature if temperature is not None else cfg.get("Temperature", 0.7)

    print(f"--- [初始化 LLM] provider={provider}, model={model}, temperature={temp} ---", flush=True)

    # ---------- OpenAI-compatible ----------
    if provider in {"openai", "openrouter", "deepseek", "qwen"}:

        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": model,
            "temperature": temp,
            "openai_api_key": api_key,
        }
        if base_url:
            kwargs["openai_api_base"] = base_url

        llm = ChatOpenAI(**kwargs)

    # ---------- Gemini ----------
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            google_api_key=api_key
        )

    else:
        raise ValueError(f"未知 Provider: {provider}")

    # 只有无覆盖参数时才缓存
    if model_name is None and temperature is None:
        _cached_llm = llm

    return llm
