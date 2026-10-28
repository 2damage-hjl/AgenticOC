"""Tests for LanceDB index builder.

Unit tests do NOT require the real BGE-M3 model or LanceDB.
They test flattening, loading, structural correctness, and safety checks.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prompt_construction.scripts.build_lancedb_index import (
    batch_iter,
    build_index,
    flatten_processed_record,
    load_processed_files,
    validate_embedding_texts,
    _safe_delete_db,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processed_record(**overrides) -> dict[str, Any]:
    """Create a processed record with sensible defaults."""
    base = {
        "processed_id": "Dialogue/marriage_dialogue/Abigail:Bad_7:en",
        "canonical_id": "Dialogue/marriage_dialogue/Abigail:Bad_7",
        "author_key": "Bad_7",
        "character": "Abigail",
        "dialogue_type": "marriage_dialogue",
        "lang": "en",
        "text": {
            "raw": "I used to be special...$s",
            "display": "I used to be special...",
            "embedding": "I used to be special...",
        },
        "context": {
            "source_key": "Bad_7",
            "key_type": "marriage_dialogue",
            "special_key_type": "marriage_mood_bad",
            "season": ["any"],
            "day_of_week": ["any"],
            "day_of_month": ["any"],
            "weather": ["any"],
            "time_period": ["any"],
            "scene": "any",
            "heart_min": None,
            "relationship_gate": "married_to_Abigail",
            "variant_index": 7,
        },
        "control": {
            "required_flags": ["relationship.married_to.Abigail"],
            "route": "any",
        },
        "quality": {
            "source_confidence": "high",
            "script_parse_confidence": "high",
        },
        "indexing": {
            "text_hash": "abc123",
            "retrieval_text_hash": "def456",
            "embedding_model": "BAAI/bge-m3",
            "vector_ref": "Dialogue/marriage_dialogue/Abigail:Bad_7:en:text",
            "_embedding_text": "Character: Abigail\nDialogue type: marriage_dialogue\nDialogue: I used to be special...",
        },
    }
    base.update(overrides)
    return base


# ===================================================================
# Test: Flattening
# ===================================================================

class TestFlattenProcessedRecord:

    def test_processed_id_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["processed_id"] == "Dialogue/marriage_dialogue/Abigail:Bad_7:en"

    def test_canonical_id_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["canonical_id"] == "Dialogue/marriage_dialogue/Abigail:Bad_7"

    def test_author_key_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["author_key"] == "Bad_7"

    def test_character_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["character"] == "Abigail"

    def test_dialogue_type_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["dialogue_type"] == "marriage_dialogue"

    def test_lang_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["lang"] == "en"

    def test_text_display_from_text_display(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["text_display"] == "I used to be special..."

    def test_text_embedding_from_text_embedding(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["text_embedding"] == "I used to be special..."

    def test_embedding_text_from_indexing(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert "Character: Abigail" in row["embedding_text"]
        assert "marriage_dialogue" in row["embedding_text"]

    def test_required_flags_as_pipe_string(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["required_flags"] == "relationship.married_to.Abigail"

    def test_required_flags_json(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        parsed = json.loads(row["required_flags_json"])
        assert "relationship.married_to.Abigail" in parsed

    def test_route_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["route"] == "any"

    def test_season_joined(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["season"] == "any"

    def test_weather_joined(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["weather"] == "any"

    def test_heart_min_none(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["heart_min"] is None

    def test_relationship_gate_marriage(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["relationship_gate"] == "married_to_Abigail"

    def test_variant_index(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["variant_index"] == 7

    def test_vector_ref_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["vector_ref"] == "Dialogue/marriage_dialogue/Abigail:Bad_7:en:text"

    def test_text_hash_preserved(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["text_hash"] == "abc123"

    def test_source_confidence(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert row["source_confidence"] == "high"

    def test_empty_required_flags(self):
        rec = _make_processed_record(control={"required_flags": [], "route": "any"})
        row = flatten_processed_record(rec)
        assert row["required_flags"] == ""
        assert json.loads(row["required_flags_json"]) == []

    def test_multiple_required_flags_pipe_delimited(self):
        rec = _make_processed_record(
            control={
                "required_flags": ["relationship.married_to.Abigail", "world.ginger_island.beach_resort.opened"],
                "route": "any",
            }
        )
        row = flatten_processed_record(rec)
        assert "relationship.married_to.Abigail" in row["required_flags"]
        assert "world.ginger_island.beach_resort.opened" in row["required_flags"]
        assert "|" in row["required_flags"]

    def test_required_flags_and_route_not_in_embedding_text(self):
        rec = _make_processed_record()
        row = flatten_processed_record(rec)
        assert "required_flags" not in row["embedding_text"]
        assert "route" not in row["embedding_text"].lower() or not any(
            line.startswith("Route:") for line in row["embedding_text"].split("\n")
        )


# ===================================================================
# Test: Loading processed files
# ===================================================================

class TestLoadProcessedFiles:

    def test_load_multiple_files(self):
        """Multiple *_processed.json files are loaded and merged."""
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            # Create two fake processed files
            file1 = {
                "dataset_meta": {"schema": "processed_dialogue_v1", "character": "Abigail"},
                "records": [_make_processed_record(character="Abigail")],
            }
            file2 = {
                "dataset_meta": {"schema": "processed_dialogue_v1", "character": "Maru"},
                "records": [_make_processed_record(character="Maru")],
            }
            (td / "Abigail_processed.json").write_text(
                json.dumps(file1, ensure_ascii=False), encoding="utf-8"
            )
            (td / "Maru_processed.json").write_text(
                json.dumps(file2, ensure_ascii=False), encoding="utf-8"
            )

            records = load_processed_files(td)
            assert len(records) == 2
            characters = {r["character"] for r in records}
            assert characters == {"Abigail", "Maru"}

    def test_no_processed_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            records = load_processed_files(Path(d))
            assert records == []

    def test_records_from_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            rec1 = _make_processed_record(processed_id="id1", lang="en")
            rec2 = _make_processed_record(processed_id="id2", lang="zh")
            file_data = {
                "dataset_meta": {"schema": "processed_dialogue_v1"},
                "records": [rec1, rec2],
            }
            (td / "Test_processed.json").write_text(
                json.dumps(file_data, ensure_ascii=False), encoding="utf-8"
            )
            records = load_processed_files(td)
            assert len(records) == 2


# ===================================================================
# Test: Validation
# ===================================================================

class TestValidateEmbeddingTexts:

    def test_valid_rows_pass(self):
        rows = [
            {"processed_id": "a:en", "embedding_text": "Character: A\nDialogue: hello"},
            {"processed_id": "b:en", "embedding_text": "Character: B\nDialogue: bye"},
        ]
        validate_embedding_texts(rows)  # should not raise

    def test_empty_embedding_text_raises(self):
        rows = [
            {"processed_id": "a:en", "embedding_text": "Character: A"},
            {"processed_id": "b:en", "embedding_text": ""},
            {"processed_id": "c:en", "embedding_text": "   "},
        ]
        with pytest.raises(ValueError, match="empty embedding_text"):
            validate_embedding_texts(rows)

    def test_empty_embedding_text_includes_example_ids(self):
        rows = [
            {"processed_id": "bad_1:en", "embedding_text": ""},
            {"processed_id": "bad_2:en", "embedding_text": "  "},
        ]
        with pytest.raises(ValueError, match="bad_1:en"):
            validate_embedding_texts(rows)


# ===================================================================
# Test: Safe delete guard
# ===================================================================

class TestSafeDeleteGuard:

    def test_safe_path_under_indexes_lancedb(self):
        # Should not raise for a path under .../indexes/lancedb
        safe_path = _PROJECT_ROOT / "prompt_construction" / "data" / "indexes" / "lancedb"
        _safe_delete_db(safe_path)  # should not raise

    def test_suspicious_path_refuses_deletion(self):
        with pytest.raises(ValueError, match="Refusing"):
            _safe_delete_db(Path("C:/Windows/System32"))

    def test_path_without_indexes_refuses(self):
        with pytest.raises(ValueError, match="Refusing"):
            _safe_delete_db(_PROJECT_ROOT / "some" / "random" / "path")

    def test_path_outside_project_root_refuses(self):
        # A path with indexes/lancedb but outside project root
        outside = Path("/tmp/some/other/project/data/indexes/lancedb")
        with pytest.raises(ValueError, match="Refusing"):
            _safe_delete_db(outside)


# ===================================================================
# Test: Batch iterator
# ===================================================================

class TestBatchIter:

    def test_batch_iter_yields_correct_batches(self):
        items = list(range(10))
        batches = list(batch_iter(items, 3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_batch_iter_does_not_materialize_all_at_once(self):
        """batch_iter should be a generator, not a list."""
        items = list(range(100))
        gen = batch_iter(items, 10)
        # Should be a generator
        import types
        assert isinstance(gen, types.GeneratorType)

    def test_batch_iter_empty_list(self):
        batches = list(batch_iter([], 5))
        assert batches == []

    def test_batch_iter_single_item(self):
        batches = list(batch_iter([1], 5))
        assert batches == [[1]]


# ===================================================================
# Test: Build index with limit and mocked model
# ===================================================================

class TestBuildIndexWithLimit:

    def test_build_index_limit_mode(self):
        """build_index with limit should only index first N records using a mocked model."""
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            processed_dir = td / "processed"
            processed_dir.mkdir()
            db_path = td / "indexes" / "lancedb"

            # Create a small processed file
            recs = [
                _make_processed_record(processed_id=f"id_{i}:en", lang="en", character=f"Char{i}")
                for i in range(10)
            ]
            file_data = {"dataset_meta": {"schema": "processed_dialogue_v1"}, "records": recs}
            (processed_dir / "Test_processed.json").write_text(
                json.dumps(file_data, ensure_ascii=False), encoding="utf-8"
            )

            # Mock SentenceTransformer and lancedb
            mock_model = MagicMock()
            dim = 8
            import numpy as np
            mock_model.encode.return_value = np.random.rand(3, dim).astype("float32")

            with patch("prompt_construction.scripts.build_lancedb_index.SentenceTransformer", return_value=mock_model):
                with patch("prompt_construction.scripts.build_lancedb_index.lancedb") as mock_lancedb:
                    mock_db = MagicMock()
                    mock_lancedb.connect.return_value = mock_db
                    mock_table = MagicMock()
                    mock_db.create_table.return_value = mock_table

                    result = build_index(
                        processed_dir=processed_dir,
                        db_path=db_path,
                        limit=3,
                    )

            assert result["records_indexed"] == 3
            mock_db.create_table.assert_called_once()

    def test_build_index_normalize_embeddings_config(self):
        """normalize_embeddings=None should use config value."""
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            processed_dir = td / "processed"
            processed_dir.mkdir()
            db_path = td / "indexes" / "lancedb"

            recs = [_make_processed_record(processed_id="id_0:en")]
            file_data = {"dataset_meta": {"schema": "processed_dialogue_v1"}, "records": recs}
            (processed_dir / "Test_processed.json").write_text(
                json.dumps(file_data, ensure_ascii=False), encoding="utf-8"
            )

            # Write a config with normalize_embeddings: false
            config_dir = td / "configs"
            config_dir.mkdir()
            config_path = config_dir / "index_config.yaml"
            config_path.write_text(
                "embedding:\n  model_name: BAAI/bge-m3\n  batch_size: 4\n  normalize_embeddings: false\n",
                encoding="utf-8",
            )

            mock_model = MagicMock()
            import numpy as np
            mock_model.encode.return_value = np.random.rand(1, 8).astype("float32")

            with patch("prompt_construction.scripts.build_lancedb_index.SentenceTransformer", return_value=mock_model):
                with patch("prompt_construction.scripts.build_lancedb_index.lancedb") as mock_lancedb:
                    with patch("prompt_construction.scripts.build_lancedb_index.CONFIG_PATH", config_path):
                        mock_db = MagicMock()
                        mock_lancedb.connect.return_value = mock_db
                        mock_db.create_table.return_value = MagicMock()

                        build_index(
                            processed_dir=processed_dir,
                            db_path=db_path,
                            normalize_embeddings=None,
                        )

            # Check that normalize_embeddings=False was passed to model.encode
            call_kwargs = mock_model.encode.call_args
            assert call_kwargs.kwargs.get("normalize_embeddings") is False

    def test_build_index_normalize_embeddings_explicit_overrides_config(self):
        """Explicit normalize_embeddings should override config."""
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            processed_dir = td / "processed"
            processed_dir.mkdir()
            db_path = td / "indexes" / "lancedb"

            recs = [_make_processed_record(processed_id="id_0:en")]
            file_data = {"dataset_meta": {"schema": "processed_dialogue_v1"}, "records": recs}
            (processed_dir / "Test_processed.json").write_text(
                json.dumps(file_data, ensure_ascii=False), encoding="utf-8"
            )

            config_dir = td / "configs"
            config_dir.mkdir()
            config_path = config_dir / "index_config.yaml"
            config_path.write_text(
                "embedding:\n  model_name: BAAI/bge-m3\n  batch_size: 4\n  normalize_embeddings: false\n",
                encoding="utf-8",
            )

            mock_model = MagicMock()
            import numpy as np
            mock_model.encode.return_value = np.random.rand(1, 8).astype("float32")

            with patch("prompt_construction.scripts.build_lancedb_index.SentenceTransformer", return_value=mock_model):
                with patch("prompt_construction.scripts.build_lancedb_index.lancedb") as mock_lancedb:
                    with patch("prompt_construction.scripts.build_lancedb_index.CONFIG_PATH", config_path):
                        mock_db = MagicMock()
                        mock_lancedb.connect.return_value = mock_db
                        mock_db.create_table.return_value = MagicMock()

                        build_index(
                            processed_dir=processed_dir,
                            db_path=db_path,
                            normalize_embeddings=True,
                        )

            call_kwargs = mock_model.encode.call_args
            assert call_kwargs.kwargs.get("normalize_embeddings") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
