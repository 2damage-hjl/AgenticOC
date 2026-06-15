"""Query LanceDB for dialogue candidates.

Wraps vector search with game-state filtering delegated to ``filters.py``.
LanceDB retrieval returns candidates only — this is NOT a final few-shot
selector.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PKG_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = _PKG_DIR / "configs" / "index_config.yaml"

# ---------------------------------------------------------------------------
# HF mirror & offline resilience
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Prefer offline mode if model is already cached — avoids 429 / SSL errors.
# Can be overridden by setting TRANSFORMERS_OFFLINE=0 in environment.
if os.environ.get("TRANSFORMERS_OFFLINE") is None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import yaml
import lancedb

from prompt_construction.retrieval.filters import (
    DEFAULT_MIN_RESULTS,
    post_filter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_TABLE = "stardew_dialogues"
DEFAULT_DB_PATH = _PKG_DIR / "data" / "indexes" / "lancedb"
DEFAULT_SEARCH_K = 40
DEFAULT_TOP_K = 10

# ---------------------------------------------------------------------------
# Embedding model singleton (module-level cache)
# ---------------------------------------------------------------------------
# Thread-safe lazy singleton: first call loads the model, subsequent calls
# reuse it.  Supports local model path via:
#   1. index_config.yaml  →  embedding.local_path
#   2. env var             →  BGE_M3_LOCAL_PATH
# Falls back to HuggingFace hub name (e.g. "BAAI/bge-m3").
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_cached_model = None          # type: SentenceTransformer | None
_cached_model_name = ""       # the name/path used to load _cached_model


def _load_embedding_model(model_name_or_path: str):
    """Load (or return cached) SentenceTransformer model.

    Resolution order for *model_name_or_path*:
    1. If it points to an existing local directory → load directly.
    2. Else pass to SentenceTransformer, which checks HF cache then hub.
    """
    global _cached_model, _cached_model_name

    with _model_lock:
        # Fast path: same model already loaded
        if _cached_model is not None and _cached_model_name == model_name_or_path:
            return _cached_model

        from sentence_transformers import SentenceTransformer

        resolved = model_name_or_path

        # Check config / env for a local path override
        config = {}
        if CONFIG_PATH.exists():
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

        local_path = (
            os.environ.get("BGE_M3_LOCAL_PATH")
            or config.get("embedding", {}).get("local_path")
        )
        if local_path and Path(local_path).is_dir():
            resolved = local_path
            logger.info(f"[embedding] Using local model path: {resolved}")
        else:
            # Try to resolve HF cache path automatically
            resolved = _resolve_hf_cache_path(model_name_or_path) or model_name_or_path

        logger.info(f"[embedding] Loading SentenceTransformer: {resolved}")
        try:
            model = SentenceTransformer(resolved)
        except Exception as e:
            logger.warning(f"[embedding] Failed to load model '{resolved}': {e}")
            # If offline mode caused the failure, try once with online mode
            if os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                logger.info("[embedding] Retrying with TRANSFORMERS_OFFLINE=0 ...")
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                try:
                    model = SentenceTransformer(resolved)
                except Exception as e2:
                    logger.error(f"[embedding] Retry also failed: {e2}")
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # restore
                    raise
                os.environ["TRANSFORMERS_OFFLINE"] = "1"  # restore after success
            else:
                raise

        _cached_model = model
        _cached_model_name = model_name_or_path
        logger.info(f"[embedding] Model loaded and cached: {resolved}")
        return model


def _resolve_hf_cache_path(model_name: str) -> str | None:
    """Try to resolve a HuggingFace hub model name to a local snapshot path.

    Returns None if no cached snapshot is found (caller should fall back to
    the hub name and let SentenceTransformer handle it).
    """
    try:
        # Standard HF cache layout: ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<rev>/
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        # Convert "BAAI/bge-m3" → "models--BAAI--bge-m3"
        folder_name = "models--" + model_name.replace("/", "--")
        model_dir = cache_root / folder_name
        if not model_dir.is_dir():
            return None
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        # Pick the latest snapshot (by modification time)
        candidates = sorted(
            snapshots_dir.iterdir(),
            key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
            reverse=True,
        )
        for snap in candidates:
            if snap.is_dir() and (snap / "config.json").exists():
                logger.info(f"[embedding] Resolved HF cache: {snap}")
                return str(snap)
    except Exception as e:
        logger.debug(f"[embedding] HF cache resolution failed: {e}")
    return None


def get_embedding_model(model_name: str | None = None) -> "SentenceTransformer":
    """Public API: get the shared embedding model instance.

    Loads on first call, returns cached instance afterwards.
    Thread-safe.
    """
    config = {}
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    _model_name = model_name or config.get("embedding", {}).get("model_name", DEFAULT_MODEL)
    return _load_embedding_model(_model_name)


def reset_embedding_model() -> None:
    """Force re-load of the embedding model on next call.

    Mainly useful for tests or config hot-reload scenarios.
    """
    global _cached_model, _cached_model_name
    with _model_lock:
        _cached_model = None
        _cached_model_name = ""


def search_dialogues(
    query_text: str,
    target_lang: str | None = None,
    game_flags: set[str] | None = None,
    route_state: str = "community_center_completed",
    top_k: int = DEFAULT_TOP_K,
    search_k: int = DEFAULT_SEARCH_K,
    db_path: Path | None = None,
    table_name: str | None = None,
    model_name: str | None = None,
    min_lang_results: int = DEFAULT_MIN_RESULTS,
    *,
    character: str | None = None,
    season: str | None = None,
    weather: str | None = None,
    day_of_week: str | None = None,
    day_of_month: str | None = None,
    heart_level: int | None = None,
    relationship: str | None = None,
) -> list[dict]:
    """Search LanceDB for dialogue candidates.

    Flow:
    1. Embed query_text (using cached model)
    2. Vector search with search_k (pre-filter character via LanceDB where)
    3. Filter by context fields (season, weather, heart_level, relationship, …)
    4. Apply route visibility
    5. Apply required_flags visibility
    6. Apply language filter with fallback (target_lang → en → any)
    7. Return top_k

    Args:
        query_text: Text to search for similar dialogues.
        target_lang: Preferred output language (e.g. "zh", "en").
        game_flags: Set of currently active game flags.
        route_state: Current community center / Joja route state.
        top_k: Final number of results to return.
        search_k: Number of candidates from vector search (before filtering).
        db_path: Override LanceDB path.
        table_name: Override table name.
        model_name: Override embedding model name.
        min_lang_results: Minimum results needed in target_lang before
            triggering fallback to en → any.  Records from fallback rounds
            carry a ``fallback_reason`` field ("fallback_en" / "fallback_any").
        character: Filter by NPC character name (e.g. "Abigail").
        season: Filter by season (e.g. "spring", "summer", "fall", "winter").
        weather: Filter by weather (e.g. "rain", "sun", "storm").
        day_of_week: Filter by day of week (e.g. "Mon", "Tue").
        day_of_month: Filter by day of month (e.g. "1", "15").
        heart_level: Player's current heart level with this character;
            records with heart_min > heart_level are excluded.
        relationship: Relationship gate (e.g. "heart_min_4", "married").

    Returns list of LanceDB result dicts.
    """
    game_flags = game_flags or set()

    # Load config
    config = {}
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

    _db_path = db_path or _PKG_DIR / config.get("vector_db_path", str(DEFAULT_DB_PATH.relative_to(_PKG_DIR)))
    _table_name = table_name or config.get("table_name", DEFAULT_TABLE)
    _search_k = search_k or int(config.get("defaults", {}).get("search_k", DEFAULT_SEARCH_K))
    _top_k = top_k or int(config.get("defaults", {}).get("top_k", DEFAULT_TOP_K))

    # Embed query — use cached model (no repeated loading)
    model = get_embedding_model(model_name)
    query_vector = model.encode(
        query_text,
        normalize_embeddings=bool(config.get("embedding", {}).get("normalize_embeddings", True)),
    ).tolist()

    # Search — use LanceDB where clause for character pre-filter
    db = lancedb.connect(str(_db_path))
    table = db.open_table(_table_name)

    search_builder = table.search(query_vector).limit(_search_k)
    if character is not None:
        search_builder = search_builder.where(f'character = "{character}"', prefilter=True)

    raw_results = search_builder.to_list()

    # Post-filter
    visible = post_filter(
        raw_results,
        game_flags,
        route_state,
        target_lang,
        min_lang_results=min_lang_results,
        character=character,
        season=season,
        weather=weather,
        day_of_week=day_of_week,
        day_of_month=day_of_month,
        heart_level=heart_level,
        relationship=relationship,
    )

    return visible[:_top_k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a sample query from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Query Stardew dialogue LanceDB")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--lang", default=None, help="Target language (e.g. zh, en)")
    parser.add_argument("--route", default="community_center_completed", help="Route state")
    parser.add_argument("--flags", default="", help="Comma-separated game flags")
    parser.add_argument("--character", default=None, help="NPC character name (e.g. Abigail)")
    parser.add_argument("--season", default=None, help="Season filter (spring/summer/fall/winter)")
    parser.add_argument("--weather", default=None, help="Weather filter (rain/sun/storm)")
    parser.add_argument("--day-of-week", default=None, help="Day of week (Mon/Tue/…)")
    parser.add_argument("--day-of-month", default=None, help="Day of month (1-28)")
    parser.add_argument("--heart-level", type=int, default=None, help="Player heart level")
    parser.add_argument("--relationship", default=None, help="Relationship gate (e.g. heart_min_4, married)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--min-lang", type=int, default=3, help="Min target-lang results before fallback")
    args = parser.parse_args()

    flags = set(f.strip() for f in args.flags.split(",") if f.strip()) if args.flags else set()

    results = search_dialogues(
        query_text=args.query,
        target_lang=args.lang,
        game_flags=flags,
        route_state=args.route,
        top_k=args.top_k,
        min_lang_results=args.min_lang,
        character=args.character,
        season=args.season,
        weather=args.weather,
        day_of_week=args.day_of_week,
        day_of_month=args.day_of_month,
        heart_level=args.heart_level,
        relationship=args.relationship,
    )

    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Rank {i}")
        print(f"ID: {r.get('processed_id', '')}")
        print(f"Character: {r.get('character', '')}")
        print(f"Type: {r.get('dialogue_type', '')}")
        print(f"Lang: {r.get('lang', '')}")
        print(f"Text: {r.get('text_display', '')}")
        print(f"Route: {r.get('route', '')}")
        print(f"Required: {r.get('required_flags', '')}")
        print(f"Relationship: {r.get('relationship_gate', '')}")
        print(f"Distance: {r.get('_distance', '')}")
        fb = r.get("fallback_reason")
        if fb:
            print(f"Fallback: {fb}")


if __name__ == "__main__":
    main()
