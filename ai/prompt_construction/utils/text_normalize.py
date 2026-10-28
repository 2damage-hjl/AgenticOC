import re


def normalize_dialogue_text(text: str | None) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text