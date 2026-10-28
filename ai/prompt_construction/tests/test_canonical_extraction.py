"""Tests for canonical extraction and multilingual alignment."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dialogue_key_parser import parse_dialogue_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dialogue_file(
    tmp: Path, name: str, data: dict[str, str]
) -> Path:
    """Write a dialogue JSON file and return its path."""
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Multilingual alignment tests
# ---------------------------------------------------------------------------

class TestMultilingualAlignment:
    """Test that multilingual texts merge under the same canonical_id."""

    def test_same_character_key_merges(self):
        """Same author_key under the same character across en and zh merges
        into one canonical record with both language texts."""
        from scripts.extract_dialogue_canonical import (
            build_record,
            make_canonical_id,
        )

        character = "Abigail"
        author_key = "winter_Wed2"
        dialogue_type = "general_dialogue"
        canonical_id = make_canonical_id(dialogue_type, character, author_key)

        parsed = parse_dialogue_key(author_key, is_marriage=False)

        # Build record for English
        rec_en = build_record(
            canonical_id=canonical_id,
            author_key=author_key,
            character=character,
            dialogue_type=dialogue_type,
            parsed=parsed,
            text_lang="en",
            text="English text",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )

        # Simulate merging zh text
        rec_en["texts"]["zh"] = "中文文本"
        rec_en["source"]["files"]["zh"] = "data/game_scripts/Dialogue/Abigail.zh-CN.json"

        assert "en" in rec_en["texts"]
        assert "zh" in rec_en["texts"]
        assert rec_en["texts"]["en"] == "English text"
        assert rec_en["texts"]["zh"] == "中文文本"

    def test_different_characters_same_key_no_collision(self):
        """Same author_key under different characters produces different canonical_ids."""
        from scripts.extract_dialogue_canonical import make_canonical_id

        id_abigail = make_canonical_id("general_dialogue", "Abigail", "winter_Wed2")
        id_sebastian = make_canonical_id("general_dialogue", "Sebastian", "winter_Wed2")
        assert id_abigail != id_sebastian

    def test_general_vs_marriage_same_key_no_collision(self):
        """Same author_key under different dialogue types produces different canonical_ids."""
        from scripts.extract_dialogue_canonical import make_canonical_id

        id_general = make_canonical_id("general_dialogue", "Abigail", "Mon")
        id_marriage = make_canonical_id("marriage_dialogue", "Abigail", "Mon")
        assert id_general != id_marriage


# ---------------------------------------------------------------------------
# Marriage required_flags tests
# ---------------------------------------------------------------------------

class TestMarriageRequiredFlags:
    """Marriage records must include relationship.married_to.{character}."""

    def test_marriage_record_has_required_flag(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        character = "Abigail"
        dialogue_type = "marriage_dialogue"
        canonical_id = make_canonical_id(dialogue_type, character, "Rainy_Day_0")
        parsed = parse_dialogue_key("Rainy_Day_0", is_marriage=True)

        record = build_record(
            canonical_id=canonical_id,
            author_key="Rainy_Day_0",
            character=character,
            dialogue_type=dialogue_type,
            parsed=parsed,
            text_lang="en",
            text="The dark... the rain...",
            file_path=Path("data/game_scripts/Dialogue/MarriageDialogueAbigail.json"),
        )

        assert "relationship.married_to.Abigail" in record["control"]["required_flags"]

    def test_general_record_no_marriage_flag(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        character = "Abigail"
        dialogue_type = "general_dialogue"
        canonical_id = make_canonical_id(dialogue_type, character, "Mon")
        parsed = parse_dialogue_key("Mon", is_marriage=False)

        record = build_record(
            canonical_id=canonical_id,
            author_key="Mon",
            character=character,
            dialogue_type=dialogue_type,
            parsed=parsed,
            text_lang="en",
            text="Oh, hey.",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )

        assert "relationship.married_to.Abigail" not in record["control"]["required_flags"]

    def test_resort_key_adds_required_flag(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        character = "Abigail"
        dialogue_type = "general_dialogue"
        canonical_id = make_canonical_id(dialogue_type, character, "Resort")
        parsed = parse_dialogue_key("Resort", is_marriage=False)

        record = build_record(
            canonical_id=canonical_id,
            author_key="Resort",
            character=character,
            dialogue_type=dialogue_type,
            parsed=parsed,
            text_lang="en",
            text="Island text",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )

        assert "world.ginger_island.beach_resort.opened" in record["control"]["required_flags"]


# ---------------------------------------------------------------------------
# Marriage relationship_gate in canonical records
# ---------------------------------------------------------------------------

class TestMarriageRelationshipGateInRecord:
    """Marriage records must have heart_min=10 and relationship_gate=married."""

    def test_marriage_record_heart_min_10(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        parsed = parse_dialogue_key("Rainy_Day_2", is_marriage=True)
        record = build_record(
            canonical_id=make_canonical_id("marriage_dialogue", "Abigail", "Rainy_Day_2"),
            author_key="Rainy_Day_2",
            character="Abigail",
            dialogue_type="marriage_dialogue",
            parsed=parsed,
            text_lang="en",
            text="Rain text",
            file_path=Path("data/game_scripts/Dialogue/MarriageDialogueAbigail.json"),
        )
        assert record["script_conditions"]["heart_min"] == 10
        assert record["script_conditions"]["relationship_gate"] == "married"

    def test_marriage_variant_index_in_record(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        parsed = parse_dialogue_key("Good_1", is_marriage=True)
        record = build_record(
            canonical_id=make_canonical_id("marriage_dialogue", "Abigail", "Good_1"),
            author_key="Good_1",
            character="Abigail",
            dialogue_type="marriage_dialogue",
            parsed=parsed,
            text_lang="en",
            text="Good mood",
            file_path=Path("data/game_scripts/Dialogue/MarriageDialogueAbigail.json"),
        )
        assert record["script_conditions"]["variant_index"] == 1

    def test_general_record_still_uses_heart(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        parsed = parse_dialogue_key("winter_Wed2", is_marriage=False)
        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", "winter_Wed2"),
            author_key="winter_Wed2",
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parsed,
            text_lang="en",
            text="Some text",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )
        assert record["script_conditions"]["heart_min"] == 2
        assert record["script_conditions"]["relationship_gate"] == "heart_min_2"


# ---------------------------------------------------------------------------
# Dialogue type tests
# ---------------------------------------------------------------------------

class TestDialogueType:
    def test_general_dialogue_type(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", "Mon"),
            author_key="Mon",
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parse_dialogue_key("Mon"),
            text_lang="en",
            text="Hi",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )
        assert record["dialogue_type"] == "general_dialogue"

    def test_marriage_dialogue_type(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        record = build_record(
            canonical_id=make_canonical_id("marriage_dialogue", "Abigail", "Indoor_Day_0"),
            author_key="Indoor_Day_0",
            character="Abigail",
            dialogue_type="marriage_dialogue",
            parsed=parse_dialogue_key("Indoor_Day_0", is_marriage=True),
            text_lang="en",
            text="Hey",
            file_path=Path("data/game_scripts/Dialogue/MarriageDialogueAbigail.json"),
        )
        assert record["dialogue_type"] == "marriage_dialogue"


# ---------------------------------------------------------------------------
# Author key preservation
# ---------------------------------------------------------------------------

class TestAuthorKeyPreservation:
    def test_author_key_preserved(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        author_key = "winter_Wed6"
        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", author_key),
            author_key=author_key,
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parse_dialogue_key(author_key),
            text_lang="en",
            text="Some text",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )
        assert record["author_key"] == author_key


# ---------------------------------------------------------------------------
# Shared marriage key character extraction
# ---------------------------------------------------------------------------

class TestSharedMarriageKeyCharacterExtraction:
    def test_character_in_shared_key(self):
        from utils.dialogue_key_parser import extract_character_from_shared_marriage_key

        char, clean = extract_character_from_shared_marriage_key("Rainy_Day_Abigail")
        assert char == "Abigail"

    def test_no_character_in_shared_key(self):
        from utils.dialogue_key_parser import extract_character_from_shared_marriage_key

        char, clean = extract_character_from_shared_marriage_key("NoBed_0")
        assert char is None


# ---------------------------------------------------------------------------
# 2.1 Missing language tests
# ---------------------------------------------------------------------------

class TestMissingLanguages:
    """When a key exists in one language but not another, the record should
    still be created with only the available language(s), and the extraction
    report should track missing_languages."""

    def test_en_only_missing_zh(self):
        """A key that only exists in en (when zh file is also present) should
        be flagged as missing zh."""
        from scripts.extract_dialogue_canonical import (
            make_canonical_id,
            extract_records,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()
            # en has both "rain" and "Mon"
            (d / "Abigail.json").write_text(
                json.dumps({"rain": "It's raining.", "Mon": "Hello"}), encoding="utf-8"
            )
            # zh only has "Mon" — "rain" is missing
            (d / "Abigail.zh-CN.json").write_text(
                json.dumps({"Mon": "你好"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, _, report = extract_records()
            finally:
                mod.INPUT_DIR = original_input

        rain_cid = make_canonical_id("general_dialogue", "Abigail", "rain")
        assert rain_cid in report["missing_languages"]
        assert "zh" in report["missing_languages"][rain_cid]
        # The record should still exist with en only
        rain_rec = [r for r in general if r["canonical_id"] == rain_cid]
        assert len(rain_rec) == 1
        assert "en" in rain_rec[0]["texts"]
        assert "zh" not in rain_rec[0]["texts"]

    def test_zh_only_missing_en(self):
        """A key that only exists in zh (when en file is also present) should
        be flagged as missing en."""
        from scripts.extract_dialogue_canonical import (
            make_canonical_id,
            extract_records,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()
            # en has "Mon"
            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello"}), encoding="utf-8"
            )
            # zh has "Mon" and "Tue" — "Tue" is missing from en
            (d / "Abigail.zh-CN.json").write_text(
                json.dumps({"Mon": "你好", "Tue": "星期二"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, _, report = extract_records()
            finally:
                mod.INPUT_DIR = original_input

        tue_cid = make_canonical_id("general_dialogue", "Abigail", "Tue")
        assert tue_cid in report["missing_languages"]
        assert "en" in report["missing_languages"][tue_cid]
        # The record should still exist with zh only
        tue_rec = [r for r in general if r["canonical_id"] == tue_cid]
        assert len(tue_rec) == 1
        assert "zh" in tue_rec[0]["texts"]
        assert "en" not in tue_rec[0]["texts"]

    def test_extra_key_in_one_language(self):
        """If en has keys {A, B} and zh has keys {A, C}, then:
        - A: both en+zh, no missing
        - B: en only, missing zh
        - C: zh only, missing en
        All three records should exist, none should error."""
        from scripts.extract_dialogue_canonical import extract_records

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()
            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello", "Tue": "Tuesday"}), encoding="utf-8"
            )
            (d / "Abigail.zh-CN.json").write_text(
                json.dumps({"Mon": "你好", "Wed": "星期三"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, _, report = extract_records()
            finally:
                mod.INPUT_DIR = original_input

        cids = {r["canonical_id"] for r in general}
        mon_cid = "Dialogue/general_dialogue/Abigail:Mon"
        tue_cid = "Dialogue/general_dialogue/Abigail:Tue"
        wed_cid = "Dialogue/general_dialogue/Abigail:Wed"
        assert mon_cid in cids
        assert tue_cid in cids
        assert wed_cid in cids

        # Mon: no missing languages (both en+zh)
        assert mon_cid not in report["missing_languages"]
        # Tue: missing zh
        assert "zh" in report["missing_languages"].get(tue_cid, [])
        # Wed: missing en
        assert "en" in report["missing_languages"].get(wed_cid, [])


# ---------------------------------------------------------------------------
# 2.2 Duplicate key / overwrite conflict tests
# ---------------------------------------------------------------------------

class TestDuplicateKeyOverwrite:
    """When the same canonical_id + lang appears twice, the extraction should
    record a warning rather than silently overwriting."""

    def test_duplicate_canonical_id_lang_creates_warning(self):
        from scripts.extract_dialogue_canonical import extract_records

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()
            # Same key appears in two en files (edge case)
            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello from file 1"}), encoding="utf-8"
            )
            # Create a subdirectory with another en file (simulating edge case)
            sub = d / "extra"
            sub.mkdir()
            (sub / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello from file 2"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, _, report = extract_records()
            finally:
                mod.INPUT_DIR = original_input

        # There should be a duplicate_overwrite warning
        mon_cid = "Dialogue/general_dialogue/Abigail:Mon"
        dup_entries = [
            e for e in report["duplicate_overwrites"] if e["canonical_id"] == mon_cid
        ]
        assert len(dup_entries) > 0, (
            f"Expected duplicate_overwrite warning for {mon_cid}, "
            f"got: {report['duplicate_overwrites']}"
        )
        assert dup_entries[0]["lang"] == "en"

    def test_no_duplicate_warning_for_different_langs(self):
        """Same canonical_id with different languages should NOT trigger duplicate warning."""
        from scripts.extract_dialogue_canonical import extract_records

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()
            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello"}), encoding="utf-8"
            )
            (d / "Abigail.zh-CN.json").write_text(
                json.dumps({"Mon": "你好"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                _, _, report = extract_records()
            finally:
                mod.INPUT_DIR = original_input

        mon_cid = "Dialogue/general_dialogue/Abigail:Mon"
        dup_entries = [
            e for e in report["duplicate_overwrites"] if e["canonical_id"] == mon_cid
        ]
        assert len(dup_entries) == 0, "Different languages should not trigger duplicate warning"


# ---------------------------------------------------------------------------
# 2.3 Source files path stability tests
# ---------------------------------------------------------------------------

class TestSourceFilePathStability:
    """source.files[lang] must be a stable string regardless of path format."""

    def test_relative_path(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", "Mon"),
            author_key="Mon",
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parse_dialogue_key("Mon"),
            text_lang="en",
            text="Hi",
            file_path=Path("data/game_scripts/Dialogue/Abigail.json"),
        )
        assert isinstance(record["source"]["files"]["en"], str)
        # Normalize separators for comparison
        assert record["source"]["files"]["en"].replace("\\", "/") == (
            "data/game_scripts/Dialogue/Abigail.json"
        )

    def test_absolute_path(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        abs_path = Path("D:/DamonAI/ai/prompt_construction/data/game_scripts/Dialogue/Abigail.json")
        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", "Mon"),
            author_key="Mon",
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parse_dialogue_key("Mon"),
            text_lang="en",
            text="Hi",
            file_path=abs_path,
        )
        assert isinstance(record["source"]["files"]["en"], str)
        # Should be relative to PROJECT_ROOT
        assert "Abigail.json" in record["source"]["files"]["en"]

    def test_windows_style_path(self):
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        # Windows backslash path
        win_path = Path("data\\game_scripts\\Dialogue\\Abigail.json")
        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", "Mon"),
            author_key="Mon",
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parse_dialogue_key("Mon"),
            text_lang="en",
            text="Hi",
            file_path=win_path,
        )
        assert isinstance(record["source"]["files"]["en"], str)
        # Should not crash and should contain the filename
        assert "Abigail.json" in record["source"]["files"]["en"]

    def test_path_does_not_crash_on_nonexistent_root(self):
        """An absolute path that is NOT under PROJECT_ROOT should still produce
        a usable string (fall back to the path as-is)."""
        from scripts.extract_dialogue_canonical import build_record, make_canonical_id

        # Absolute path that is NOT under PROJECT_ROOT
        weird_path = Path("Z:/nonexistent/Dialogue/Abigail.json")
        record = build_record(
            canonical_id=make_canonical_id("general_dialogue", "Abigail", "Mon"),
            author_key="Mon",
            character="Abigail",
            dialogue_type="general_dialogue",
            parsed=parse_dialogue_key("Mon"),
            text_lang="en",
            text="Hi",
            file_path=weird_path,
        )
        # Should not crash — either relative_to worked or fallback to str(path)
        assert isinstance(record["source"]["files"]["en"], str)


# ---------------------------------------------------------------------------
# Per-character file merging tests
# ---------------------------------------------------------------------------

class TestPerCharacterFileMerging:
    """Each {Character}_canonical.json must contain both general_dialogue
    and marriage_dialogue records for that character."""

    def test_character_file_contains_both_types(self):
        """A character with both general and marriage dialogues should have
        them merged into a single {Character}_canonical.json."""
        from scripts.extract_dialogue_canonical import (
            extract_records,
            _write_per_character_files,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()

            # General dialogue for Abigail (en)
            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello", "Tue": "Tuesday"}), encoding="utf-8"
            )
            # Marriage dialogue for Abigail (en)
            (d / "MarriageDialogueAbigail.json").write_text(
                json.dumps({"Rainy_Day_0": "Rain text", "Good_1": "Good mood"}),
                encoding="utf-8",
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, marriage, _ = extract_records()
            finally:
                mod.INPUT_DIR = original_input

            out_dir = tmp / "canonical"
            out_dir.mkdir()
            _write_per_character_files(general + marriage, out_dir)

            # Should produce Abigail_canonical.json
            char_file = out_dir / "Abigail_canonical.json"
            assert char_file.exists(), f"Expected {char_file} to exist, found: {list(out_dir.iterdir())}"

            data = json.loads(char_file.read_text(encoding="utf-8"))
            assert data["dataset_meta"]["character"] == "Abigail"
            assert data["dataset_meta"]["record_count"] == 4  # 2 general + 2 marriage

            types_in_file = {r["dialogue_type"] for r in data["records"]}
            assert "general_dialogue" in types_in_file
            assert "marriage_dialogue" in types_in_file

    def test_general_only_character_file(self):
        """A character with only general dialogues should still produce
        {Character}_canonical.json with only general_dialogue records."""
        from scripts.extract_dialogue_canonical import (
            extract_records,
            _write_per_character_files,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()

            # Only general dialogue
            (d / "Clint.json").write_text(
                json.dumps({"Mon": "Hello"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, marriage, _ = extract_records()
            finally:
                mod.INPUT_DIR = original_input

            out_dir = tmp / "canonical"
            out_dir.mkdir()
            _write_per_character_files(general + marriage, out_dir)

            char_file = out_dir / "Clint_canonical.json"
            assert char_file.exists()

            data = json.loads(char_file.read_text(encoding="utf-8"))
            types_in_file = {r["dialogue_type"] for r in data["records"]}
            assert types_in_file == {"general_dialogue"}

    def test_no_old_separate_marriage_file(self):
        """There should be NO MarriageDialogue{Character}_canonical.json
        files — marriage records are merged into {Character}_canonical.json."""
        from scripts.extract_dialogue_canonical import (
            extract_records,
            _write_per_character_files,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()

            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello"}), encoding="utf-8"
            )
            (d / "MarriageDialogueAbigail.json").write_text(
                json.dumps({"Rainy_Day_0": "Rain"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, marriage, _ = extract_records()
            finally:
                mod.INPUT_DIR = original_input

            out_dir = tmp / "canonical"
            out_dir.mkdir()
            _write_per_character_files(general + marriage, out_dir)

            # Old-style files must NOT exist
            old_marriage = out_dir / "MarriageDialogueAbigail_canonical.json"
            old_dialogue = out_dir / "Abigail_dialogue_canonical.json"
            assert not old_marriage.exists(), f"Old marriage file {old_marriage} should not exist"
            assert not old_dialogue.exists(), f"Old dialogue file {old_dialogue} should not exist"

    def test_dialogue_type_counts_in_meta(self):
        """dataset_meta.dialogue_type_counts should report counts per type."""
        from scripts.extract_dialogue_canonical import (
            extract_records,
            _write_per_character_files,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "Dialogue"
            d.mkdir()

            (d / "Abigail.json").write_text(
                json.dumps({"Mon": "Hello", "Tue": "Tuesday"}), encoding="utf-8"
            )
            (d / "MarriageDialogueAbigail.json").write_text(
                json.dumps({"Rainy_Day_0": "Rain text"}), encoding="utf-8"
            )

            import scripts.extract_dialogue_canonical as mod
            original_input = mod.INPUT_DIR
            mod.INPUT_DIR = d
            try:
                general, marriage, _ = extract_records()
            finally:
                mod.INPUT_DIR = original_input

            out_dir = tmp / "canonical"
            out_dir.mkdir()
            _write_per_character_files(general + marriage, out_dir)

            char_file = out_dir / "Abigail_canonical.json"
            data = json.loads(char_file.read_text(encoding="utf-8"))
            type_counts = data["dataset_meta"]["dialogue_type_counts"]
            assert type_counts.get("general_dialogue") == 2
            assert type_counts.get("marriage_dialogue") == 1


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
