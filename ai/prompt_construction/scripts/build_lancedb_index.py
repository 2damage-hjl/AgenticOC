"""Build LanceDB vector index from processed dialogue records.

Reads all ``*_processed.json`` files from ``data/processed/``, flattens each
processed record into a LanceDB-compatible row, encodes the
``indexing._embedding_text`` field with BAAI/bge-m3, and writes the result
to ``data/indexes/lancedb``.

MVP mode: overwrite — every run rebuilds the table from scratch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PKG_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = _PKG_DIR / "configs" / "index_config.yaml"

# ---------------------------------------------------------------------------
# HF mirror for China network
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

import yaml
import numpy as np
import lancedb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from prompt_construction.utils.json_io import load_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TABLE = "stardew_dialogues"
DEFAULT_DB_PATH = _PKG_DIR / "data" / "indexes" / "lancedb"
DEFAULT_PROCESSED_DIR = _PKG_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# Record flattening
# ---------------------------------------------------------------------------

def flatten_processed_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten a processed record into a LanceDB-compatible row.

    All fields are stored as simple scalar types (str, int, float, None)
    or lists of floats (vector).  List fields like season/weather are
    joined as comma-separated strings for SQL-like filtering.
    """
    ctx = rec.get("context", {})
    ctrl = rec.get("control", {})
    txt = rec.get("text", {})
    idx = rec.get("indexing", {})
    quality = rec.get("quality", {})

    required_flags = ctrl.get("required_flags", [])
    required_flags_str = "|".join(required_flags) if required_flags else ""
    required_flags_json = json.dumps(required_flags, ensure_ascii=False)

    def _arr_join(val: Any) -> str:
        if val is None:
            return "any"
        if isinstance(val, list):
            return ",".join(str(v) for v in val) if val else "any"
        return str(val)

    row: dict[str, Any] = {
        "processed_id": rec.get("processed_id", ""),
        "canonical_id": rec.get("canonical_id", ""),
        "author_key": rec.get("author_key", ""),
        "character": rec.get("character", ""),
        "dialogue_type": rec.get("dialogue_type", ""),
        "lang": rec.get("lang", ""),
        # Text fields
        "text_display": txt.get("display", ""),
        "text_embedding": txt.get("embedding", ""),
        "embedding_text": idx.get("_embedding_text", ""),
        # Vector placeholder — filled during embedding
        "vector": [],
        # Control metadata
        "route": ctrl.get("route", "any"),
        "required_flags": required_flags_str,
        "required_flags_json": required_flags_json,
        # Context fields
        "season": _arr_join(ctx.get("season")),
        "weather": _arr_join(ctx.get("weather")),
        "scene": ctx.get("scene", "any") or "any",
        "time_period": _arr_join(ctx.get("time_period")),
        "day_of_week": _arr_join(ctx.get("day_of_week")),
        "day_of_month": _arr_join(ctx.get("day_of_month")),
        "heart_min": ctx.get("heart_min"),
        "relationship_gate": ctx.get("relationship_gate", "any"),
        "scene_type": rec.get("scene_type", "daily"),
        "variant_index": ctx.get("variant_index"),
        "mood": ctx.get("mood"),
        "family_state": ctx.get("family_state"),
        "source_key": ctx.get("source_key", ""),
        # Indexing fields
        "vector_ref": idx.get("vector_ref", ""),
        "text_hash": idx.get("text_hash", ""),
        "retrieval_text_hash": idx.get("retrieval_text_hash", ""),
        # Quality
        "source_confidence": quality.get("source_confidence", ""),
        "script_parse_confidence": quality.get("script_parse_confidence", ""),
    }

    return row


# ---------------------------------------------------------------------------
# Loading processed files
# ---------------------------------------------------------------------------

def load_processed_files(processed_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all ``*_processed.json`` files from the processed directory.

    Returns a flat list of all processed records from all character files.
    """
    pdir = processed_dir or DEFAULT_PROCESSED_DIR
    records: list[dict[str, Any]] = []

    files = sorted(pdir.glob("*_processed.json"))
    if not files:
        logger.warning("No processed files found in %s", pdir)
        return records

    for fp in files:
        data = load_json(fp)
        file_records = data.get("records", [])
        records.extend(file_records)

    return records


# ---------------------------------------------------------------------------
# Batch encoding — streaming iterator
# ---------------------------------------------------------------------------

def batch_iter(items: list, batch_size: int) -> Generator[list, None, None]:
    """Yield batches of items without materializing all batches at once."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_embedding_texts(rows: list[dict[str, Any]]) -> None:
    """Raise ValueError if any row has empty embedding_text."""
    empty_ids = [row["processed_id"] for row in rows if not row.get("embedding_text", "").strip()]
    if empty_ids:
        sample = empty_ids[:10]
        raise ValueError(
            f"Found {len(empty_ids)} records with empty embedding_text. "
            f"Example processed_ids: {sample}"
        )


# ---------------------------------------------------------------------------
# Safe delete guard
# ---------------------------------------------------------------------------

def _safe_delete_db(db_path: Path) -> None:
    """Only allow deleting paths under data/indexes/lancedb.

    Raises ValueError if the resolved path is suspicious.
    """
    resolved = db_path.resolve()
    # Must be under a data/indexes/lancedb directory somewhere in the project
    # Check that the path contains "indexes" and "lancedb" as components
    parts = resolved.parts
    try:
        idx_pos = list(parts).index("indexes")
        if idx_pos + 1 < len(parts) and parts[idx_pos + 1] == "lancedb":
            # Looks safe — under .../indexes/lancedb
            pass
        else:
            raise ValueError(
                f"Refusing to delete {resolved}: path does not end with indexes/lancedb"
            )
    except ValueError:
        raise ValueError(
            f"Refusing to delete {resolved}: path does not contain 'indexes' directory"
        )

    # Extra sanity: must be under the project root
    project_root = _PROJECT_ROOT.resolve()
    if not str(resolved).startswith(str(project_root)):
        raise ValueError(
            f"Refusing to delete {resolved}: path is outside project root {project_root}"
        )


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(
    processed_dir: Path | None = None,
    db_path: Path | None = None,
    table_name: str | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    normalize_embeddings: bool | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the LanceDB index from processed files.

    Parameters
    ----------
    limit : int | None
        If provided, only index the first N processed records.
        Useful for quick testing / debugging.

    Returns a summary dict.
    """
    # Load config
    config: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    print(f"[checkpoint] Config loaded from {CONFIG_PATH}")

    _model_name = model_name or config.get("embedding", {}).get("model_name", DEFAULT_MODEL)
    _batch_size = batch_size or int(config.get("embedding", {}).get("batch_size", DEFAULT_BATCH_SIZE))
    # normalize_embeddings: explicit param overrides; None → use config; config missing → True
    if normalize_embeddings is not None:
        _normalize = normalize_embeddings
    else:
        _normalize = bool(config.get("embedding", {}).get("normalize_embeddings", True))
    _db_path = db_path or _PKG_DIR / config.get("vector_db_path", str(DEFAULT_DB_PATH.relative_to(_PKG_DIR)))
    _table_name = table_name or config.get("table_name", DEFAULT_TABLE)
    _processed_dir = processed_dir or _PKG_DIR / config.get("processed_dir", str(DEFAULT_PROCESSED_DIR.relative_to(_PKG_DIR)))
    print(f"[checkpoint] Config: model={_model_name}, batch_size={_batch_size}, normalize={_normalize}, limit={limit}")

    # Load records
    records = load_processed_files(_processed_dir)
    print(f"[checkpoint] Processed files found in {_processed_dir}, total records: {len(records)}")
    if not records:
        logger.error("No processed records found. Run build_processed_dialogues first.")
        return {"error": "no_records", "records_indexed": 0}

    # Apply limit
    if limit is not None and limit > 0:
        records = records[:limit]
        print(f"[checkpoint] Limit applied: using first {len(records)} records")

    # Flatten
    rows = [flatten_processed_record(rec) for rec in records]
    print(f"[checkpoint] Records flattened: {len(rows)} rows")

    # Validate embedding texts
    validate_embedding_texts(rows)
    print(f"[checkpoint] Embedding text validation passed")

    # Load model — reuse cached singleton from query_lancedb
    print(f"[checkpoint] Loading embedding model: {_model_name} ...")
    from prompt_construction.retrieval.query_lancedb import get_embedding_model
    model = get_embedding_model(_model_name)
    print(f"[checkpoint] Embedding model loaded: {_model_name}")

    # Encode in batches — streaming iterator
    vectors_all: list[list[float]] = []
    embedding_texts = [row["embedding_text"] for row in rows]
    total_batches = (len(embedding_texts) + _batch_size - 1) // _batch_size

    print(f"[checkpoint] Embedding started: {len(rows)} records in ~{total_batches} batches of {_batch_size}")
    for batch_idx, batch in enumerate(tqdm(batch_iter(embedding_texts, _batch_size), total=total_batches, desc="Embedding")):
        vecs = model.encode(
            batch,
            batch_size=_batch_size,
            normalize_embeddings=_normalize,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype="float32")
        vectors_all.extend(vecs.tolist())
        if (batch_idx + 1) % 50 == 0:
            print(f"[checkpoint] Embedded {(batch_idx + 1) * _batch_size}/{len(rows)} records")

    print(f"[checkpoint] Embedding completed: {len(vectors_all)} vectors")

    # Assign vectors
    for row, vec in zip(rows, vectors_all):
        row["vector"] = vec

    # Wipe and rebuild
    if _db_path.exists():
        _safe_delete_db(_db_path)
        print(f"[checkpoint] Deleted existing DB at {_db_path}")
        shutil.rmtree(_db_path)
    _db_path.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] LanceDB writing started: {_db_path} / {_table_name}")

    db = lancedb.connect(str(_db_path))
    table = db.create_table(_table_name, data=rows, mode="overwrite")
    print(f"[checkpoint] LanceDB writing completed")

    summary = {
        "records_indexed": len(rows),
        "db_path": str(_db_path),
        "table_name": _table_name,
        "model": _model_name,
    }
    logger.info("Built LanceDB: %s / %s — %d records", _db_path, _table_name, len(rows))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LanceDB index from processed dialogue records")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N records (for testing)")
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to processed directory")
    parser.add_argument("--db-path", type=str, default=None, help="Path to LanceDB database directory")
    parser.add_argument("--table-name", type=str, default=None, help="LanceDB table name")
    parser.add_argument("--model-name", type=str, default=None, help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    result = build_index(
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        db_path=Path(args.db_path) if args.db_path else None,
        table_name=args.table_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"Records indexed: {result.get('records_indexed', 0)}")
    print(f"DB path: {result.get('db_path', '')}")
    print(f"Table: {result.get('table_name', '')}")


if __name__ == "__main__":
    main()
