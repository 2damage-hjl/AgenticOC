import hashlib
from .text_normalize import normalize_dialogue_text


def make_text_hash(text: str) -> str:
    normalized = normalize_dialogue_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_retrieval_text_hash(retrieval_text: str) -> str:
    normalized = normalize_dialogue_text(retrieval_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()