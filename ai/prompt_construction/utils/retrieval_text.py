from typing import Any
from .text_normalize import normalize_dialogue_text


def _as_list_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(v) for v in value if str(v).strip())

    return str(value)


def build_retrieval_text(item: dict, default_character: str = "Abigail") -> str:
    data = item.get("data", {})
    context = item.get("context", {})

    dialogue = normalize_dialogue_text(data.get("text", ""))

    character = context.get("character", default_character)
    attitude = context.get("attitude", "")
    relationship_status = context.get("relationship_status", "")
    scene = context.get("scene", "")
    season = _as_list_text(context.get("season", []))
    weather = _as_list_text(context.get("weather", []))
    time_period = _as_list_text(context.get("time_period", []))
    topic_tags = _as_list_text(context.get("topic_tags", []))
    intent = context.get("intent", "")
    tone = context.get("tone", "")
    length_bucket = context.get("length_bucket", "")
    date = context.get("date", "")

    return "\n".join(
        [
            f"Character: {character}",
            f"Relationship: {relationship_status}",
            f"Scene: {scene}",
            f"Season: {season}",
            f"Weather: {weather}",
            f"Time: {time_period}",
            f"Date: {date}",
            f"Attitude: {attitude}",
            f"Intent: {intent}",
            f"Tone: {tone}",
            f"Topics: {topic_tags}",
            f"Length: {length_bucket}",
            f"Dialogue: {dialogue}",
        ]
    )