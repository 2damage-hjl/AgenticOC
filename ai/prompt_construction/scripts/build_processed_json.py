'''
1. 读取 raw Abigail.json；
2. 自动生成 text_hash；
3. 自动生成 retrieval_text；
4. 自动生成 retrieval_text_hash；
5. 写入 processed JSON。
'''
import sys
from pathlib import Path

# 确保 prompt_construction 包可被导入
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from prompt_construction.utils.json_io import load_json, save_json
from prompt_construction.utils.hashing import make_text_hash, make_retrieval_text_hash
from prompt_construction.utils.retrieval_text import build_retrieval_text
from prompt_construction.utils.text_normalize import normalize_dialogue_text


_PKG_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = _PKG_DIR / "configs" / "index_config.yaml"


def normalize_raw_payload(raw_data):
    """
    支持两种格式：
    1. 直接是 list[dialogue]
    2. {"dataset_meta": ..., "dialogues": [...]}
    """
    if isinstance(raw_data, list):
        return {}, raw_data

    if isinstance(raw_data, dict) and "dialogues" in raw_data:
        return raw_data.get("dataset_meta", {}), raw_data["dialogues"]

    raise ValueError("Unsupported raw JSON format. Expected list or dict with key 'dialogues'.")


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    raw_path = _PKG_DIR / config["raw_path"]
    processed_path = _PKG_DIR / config["processed_path"]
    default_character = config.get("defaults", {}).get("character", "Abigail")

    raw_data = load_json(raw_path)
    dataset_meta, dialogues = normalize_raw_payload(raw_data)

    processed_items = []

    for item in dialogues:
        if "id" not in item:
            raise ValueError(f"Dialogue item missing id: {item}")

        text = item.get("data", {}).get("text", "")
        normalized_text = normalize_dialogue_text(text)

        if not normalized_text:
            raise ValueError(f"Dialogue item has empty text: {item['id']}")

        text_hash = make_text_hash(normalized_text)
        retrieval_text = build_retrieval_text(item, default_character=default_character)
        retrieval_text_hash = make_retrieval_text_hash(retrieval_text)

        item.setdefault("meta", {})
        item["meta"]["text_hash"] = text_hash

        item["embedding"] = {
            "model": config["embedding"]["model_name"],
            "retrieval_text_hash": retrieval_text_hash,
            "vector_ref": f"{item['id']}:text",
        }

        # 下划线字段表示机器生成，业务逻辑可读但不建议人工编辑
        item["_index_text"] = retrieval_text

        processed_items.append(item)

    output = {
        "dataset_meta": {
            **dataset_meta,
            "processed_from": str(raw_path),
            "schema_stage": "processed",
            "embedding_model": config["embedding"]["model_name"],
        },
        "dialogues": processed_items,
    }

    save_json(output, processed_path)

    print(f"Processed {len(processed_items)} dialogues.")
    print(f"Saved to: {processed_path}")


if __name__ == "__main__":
    main()