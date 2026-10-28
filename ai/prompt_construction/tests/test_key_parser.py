"""Tests for the dialogue key parser."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dialogue_key_parser import parse_dialogue_key, extract_character_from_shared_marriage_key


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse(key: str, *, is_marriage: bool = False) -> dict:
    return parse_dialogue_key(key, is_marriage=is_marriage)


def _conditions(key: str, *, is_marriage: bool = False) -> dict:
    return _parse(key, is_marriage=is_marriage)["script_conditions"]


def _flags(key: str, *, is_marriage: bool = False) -> list[str]:
    return _parse(key, is_marriage=is_marriage)["key_required_flags"]


# ---------------------------------------------------------------------------
# Season + DayOfWeek + Heart
# ---------------------------------------------------------------------------

class TestSeasonDayHeart:
    def test_winter_Wed2(self):
        c = _conditions("winter_Wed2")
        assert c["season"] == ["winter"]
        assert c["day_of_week"] == ["Wed"]
        assert c["heart_min"] == 2
        assert c["relationship_gate"] == "heart_min_2"
        assert c["weather"] == ["any"]
        assert c["parse_warnings"] == []

    def test_summer_Mon8(self):
        c = _conditions("summer_Mon8")
        assert c["season"] == ["summer"]
        assert c["day_of_week"] == ["Mon"]
        assert c["heart_min"] == 8
        assert c["relationship_gate"] == "heart_min_8"

    def test_spring_13(self):
        c = _conditions("spring_13")
        assert c["season"] == ["spring"]
        assert c["day_of_month"] == ["13"]
        assert c["special_key_type"] == "date_specific_dialogue"

    def test_fall_Tue4(self):
        c = _conditions("fall_Tue4")
        assert c["season"] == ["fall"]
        assert c["day_of_week"] == ["Tue"]
        assert c["heart_min"] == 4


# ---------------------------------------------------------------------------
# Standalone weekday
# ---------------------------------------------------------------------------

class TestWeekday:
    def test_Mon(self):
        c = _conditions("Mon")
        assert c["day_of_week"] == ["Mon"]
        assert c["heart_min"] is None

    def test_Sat(self):
        c = _conditions("Sat")
        assert c["day_of_week"] == ["Sat"]

    def test_Mon2(self):
        c = _conditions("Mon2")
        assert c["day_of_week"] == ["Mon"]
        assert c["heart_min"] == 2


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

class TestWeather:
    def test_rain(self):
        c = _conditions("rain")
        assert c["weather"] == ["rainy"]
        assert c["special_key_type"] == "weather_rain"

    def test_GreenRain(self):
        c = _conditions("GreenRain")
        assert c["weather"] == ["green_rain"]
        assert c["special_key_type"] == "weather_green_rain"

    def test_green_rain_variant(self):
        c = _conditions("green_rain")
        assert c["weather"] == ["green_rain"]


# ---------------------------------------------------------------------------
# Resort
# ---------------------------------------------------------------------------

class TestResort:
    def test_Resort(self):
        c = _conditions("Resort")
        f = _flags("Resort")
        assert c["special_key_type"] == "ginger_island_resort"
        assert "world.ginger_island.beach_resort.opened" in f

    def test_Resort_Shore(self):
        c = _conditions("Resort_Shore")
        f = _flags("Resort_Shore")
        assert c["special_key_type"] == "ginger_island_resort"
        assert "world.ginger_island.beach_resort.opened" in f

    def test_Resort_wildcard(self):
        """Resort_* pattern should also trigger the required flag."""
        f = _flags("Resort_Bar")
        assert "world.ginger_island.beach_resort.opened" in f


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

class TestRelationship:
    def test_breakUp(self):
        c = _conditions("breakUp")
        assert c["special_key_type"] == "breakup"
        assert c["relationship_gate"] == "breakup"

    def test_breakup_lowercase(self):
        c = _conditions("breakup")
        assert c["relationship_gate"] == "breakup"

    def test_divorced(self):
        c = _conditions("divorced")
        assert c["special_key_type"] == "divorced"
        assert c["relationship_gate"] == "divorced"


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------

class TestUnknownKey:
    def test_unknown_key_gets_warning(self):
        c = _conditions("some_weird_key_xyz")
        assert len(c["parse_warnings"]) > 0

    def test_unknown_key_not_dropped(self):
        result = _parse("some_weird_key_xyz")
        assert "script_conditions" in result
        assert result["script_conditions"]["season"] == ["any"]


# ---------------------------------------------------------------------------
# Marriage-specific keys
# ---------------------------------------------------------------------------

class TestMarriageKeys:
    def test_Rainy_Day_0(self):
        c = _conditions("Rainy_Day_0", is_marriage=True)
        assert c["weather"] == ["rainy"]

    def test_Indoor_Day_0(self):
        c = _conditions("Indoor_Day_0", is_marriage=True)
        # No weather change for indoor
        assert c["weather"] == ["any"]

    def test_Good_1(self):
        c = _conditions("Good_1", is_marriage=True)
        assert c["special_key_type"] == "marriage_mood_good"

    def test_Bad_7(self):
        c = _conditions("Bad_7", is_marriage=True)
        assert c["special_key_type"] == "marriage_mood_bad"

    def test_OneKid_3(self):
        c = _conditions("OneKid_3", is_marriage=True)
        assert c["special_key_type"] == "marriage_one_kid"

    def test_TwoKids_0(self):
        c = _conditions("TwoKids_0", is_marriage=True)
        assert c["special_key_type"] == "marriage_two_kids"

    def test_spouseRoom(self):
        c = _conditions("spouseRoom_Abigail", is_marriage=True)
        assert c["special_key_type"] == "marriage_spouse_room"


# ---------------------------------------------------------------------------
# Marriage relationship_gate: married characters have heart_min=10
# ---------------------------------------------------------------------------

class TestMarriageRelationshipGate:
    """Marriage keys must NOT interpret trailing numbers as heart levels.
    Married characters always have heart_min >= 10."""

    def test_rainy_day_0_is_not_heart_0(self):
        c = _conditions("Rainy_Day_0", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["heart_min"] != 0

    def test_indoor_day_0_is_not_heart_0(self):
        c = _conditions("Indoor_Day_0", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"

    def test_good_1_is_not_heart_1(self):
        c = _conditions("Good_1", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["variant_index"] == 1

    def test_bad_7_is_not_heart_7(self):
        c = _conditions("Bad_7", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["variant_index"] == 7

    def test_onekid_3_variant_index(self):
        c = _conditions("OneKid_3", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["variant_index"] == 3

    def test_twoKids_0_variant_index(self):
        c = _conditions("TwoKids_0", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["variant_index"] == 0

    def test_rainy_day_variant_index(self):
        c = _conditions("Rainy_Day_2", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["variant_index"] == 2

    def test_no_trailing_number_no_variant(self):
        c = _conditions("spouseRoom_Abigail", is_marriage=True)
        assert c["heart_min"] == 10
        assert c["relationship_gate"] == "married"
        assert c["variant_index"] is None

    def test_general_key_still_uses_heart(self):
        """Non-marriage keys must still interpret heart suffix correctly."""
        c = _conditions("winter_Wed2", is_marriage=False)
        assert c["heart_min"] == 2
        assert c["relationship_gate"] == "heart_min_2"

    def test_general_Mon2_still_heart(self):
        c = _conditions("Mon2", is_marriage=False)
        assert c["heart_min"] == 2
        assert c["relationship_gate"] == "heart_min_2"


# ---------------------------------------------------------------------------
# Shared marriage key extraction
# ---------------------------------------------------------------------------

class TestSharedMarriageKeyExtraction:
    def test_Rainy_Day_Abigail(self):
        char, clean = extract_character_from_shared_marriage_key("Rainy_Day_Abigail")
        assert char == "Abigail"
        assert clean == "Rainy_Day"

    def test_spring_Haley(self):
        char, clean = extract_character_from_shared_marriage_key("spring_Haley")
        assert char == "Haley"
        assert clean == "spring"

    def test_NoBed_0_no_character(self):
        char, clean = extract_character_from_shared_marriage_key("NoBed_0")
        assert char is None
        assert clean == "NoBed_0"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
