"""Few-shot example selector — turns LanceDB candidates into high-quality examples.

Pipeline:
1. Canonical dedup  — same canonical_id in different languages → keep target_lang
2. Text dedup       — same/normalized text content dedup + near-duplicate filter
3. Scene-type filter — exclude special-event types unless matched by current state
4. Target-lang priority — prefer target_lang, fallback en, then any
5. Relationship closeness — prefer samples closer to current relationship stage
6. Low-info down-weight — short / generic sentences get a penalty
7. Scene diversity — same (dialogue_type, relationship_gate) cluster gets capped
8. Pick 3-5 final examples

Input:  list[dict] from query_lancedb.search_dialogues()
Output: list[dict] with 3-5 high-quality few-shot records
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_OUTPUT = 3
MAX_OUTPUT = 5

# Low-info penalty
SHORT_CHAR_THRESHOLD = 15      # chars below this → penalty
GENERIC_PATTERN = re.compile(
    r"^[。？！…\s]*$|^(hi|hey|hello|嗯|哦|啊|是|好|行|ok|bye)[。？！…\s]*$",
    re.IGNORECASE,
)
SHORT_PENALTY = 0.3

# Scene diversity: max examples per (dialogue_type, relationship_gate) cluster
MAX_PER_SCENE = 2

# Target-lang bonus / non-target penalty
LANG_BONUS = 0.15

# Scene-type filtering: these types are excluded unless current state matches
# (birthday, festival, marriage, special, gift)
EXCLUDED_SCENE_TYPES_DEFAULT: frozenset[str] = frozenset({
    "birthday", "festival", "marriage", "special", "gift",
})

# Relationship closeness: how much to boost records whose gate is close to
# the player's current relationship stage.  The bonus decays linearly with
# the gap between gate stage and current stage.
RELATIONSHIP_CLOSENESS_BONUS = 0.12  # max bonus when gate == current

# Penalty for gate=any records when current relationship is friend+
# gate=any records often contain stranger-level content ("我不认识你")
GATE_ANY_PENALTY = 0.10

# Near-duplicate threshold: Jaccard similarity above this → dedup
NEAR_DUPLICATE_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# Relationship stage values (shared with few_shot_provider.py)
# ---------------------------------------------------------------------------

_STAGE_VALUES: dict[str, int] = {
    "stranger":        0,
    "acquaintance":    2,
    "heart_min_2":     2,
    "heart_min_4":     4,
    "heart_min_6":     6,
    "heart_min_8":     8,
    "friend":          4,
    "close_friend":    6,
    "best_friend":     8,
    "dating":          8,
    "engaged":        10,
    "married":        12,
    "spouse":         12,
    "divorced":       10,
}


def _normalize_relationship_key(s: str) -> str:
    return s.lower().strip().replace(" ", "_")


def _gate_to_stage(gate: str) -> int:
    """Map relationship_gate to numeric stage value. Returns -1 for 'any'/unknown."""
    if not gate or gate == "any":
        return -1
    key = _normalize_relationship_key(gate)
    if key.startswith("married_to_"):
        key = "married"
    if key in _STAGE_VALUES:
        return _STAGE_VALUES[key]
    return -1


def _relationship_to_stage(relationship: str) -> int:
    """Map current relationship to numeric stage value. Returns -1 for unknown."""
    if not relationship:
        return -1
    key = _normalize_relationship_key(relationship)
    if key in _STAGE_VALUES:
        return _STAGE_VALUES[key]
    for stage_name, value in _STAGE_VALUES.items():
        if key.startswith(stage_name):
            return value
    return -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char_count(text: str | None) -> int:
    """Count non-whitespace characters."""
    if not text:
        return 0
    return len(re.sub(r"\s+", "", text))


def _is_generic(text: str | None) -> bool:
    """Heuristic: is this a low-information short utterance?"""
    if not text:
        return True
    stripped = text.strip()
    if GENERIC_PATTERN.match(stripped):
        return True
    if _char_count(stripped) < SHORT_CHAR_THRESHOLD:
        return True
    return False


def _scene_key(rec: dict) -> str:
    """Cluster key for scene diversity."""
    dtype = rec.get("dialogue_type", "")
    gate = rec.get("relationship_gate", "")
    return f"{dtype}|{gate}"


def _normalize_text_for_dedup(text: str | None) -> str:
    """Normalize text for deduplication comparison.

    Strips whitespace, punctuation, and case differences.
    """
    if not text:
        return ""
    # Lowercase
    t = text.lower()
    # Remove whitespace
    t = re.sub(r"\s+", "", t)
    # Remove common punctuation (CJK + Latin)
    t = re.sub(r"[。？！，、；：""''【】（）《》\u3000,.!?;:'\"()\[\]{}<>…\-_~·]", "", t)
    # Normalize unicode
    t = unicodedata.normalize("NFKC", t)
    return t


def _char_set(text: str) -> set[str]:
    """Get set of characters for Jaccard similarity."""
    return set(text) if text else set()


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Step 1: Canonical dedup
# ---------------------------------------------------------------------------

def _canonical_dedup(
    records: list[dict],
    target_lang: str | None,
) -> list[dict]:
    """Dedup by canonical_id: keep target_lang version when available.

    For each canonical_id, prefer: target_lang > en > first available.
    """
    if target_lang is None:
        return records

    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        cid = rec.get("canonical_id") or rec.get("processed_id", "")
        buckets[cid].append(rec)

    result: list[dict] = []
    for cid, group in buckets.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Priority: target_lang > en > first
        chosen = None
        for lang in [target_lang, "en"]:
            for rec in group:
                if rec.get("lang") == lang:
                    chosen = rec
                    break
            if chosen:
                break
        result.append(chosen or group[0])

    return result


# ---------------------------------------------------------------------------
# Step 2: Text dedup (exact + near-duplicate)
# ---------------------------------------------------------------------------

def _text_dedup(records: list[dict]) -> list[dict]:
    """Remove records with duplicate or near-duplicate text_display.

    Dedup rules (applied in order):
    1. Same normalized text → keep first (highest rank)
    2. Near-duplicate (Jaccard > threshold) → keep first
    """
    seen_normalized: list[tuple[str, set[str]]] = []  # (normalized, char_set)
    result: list[dict] = []

    for rec in records:
        text = rec.get("text_display") or rec.get("text_raw") or ""
        normalized = _normalize_text_for_dedup(text)

        if not normalized:
            continue

        char_set = _char_set(normalized)

        # Check against all seen texts
        is_dup = False
        for prev_norm, prev_set in seen_normalized:
            # Exact match on normalized form
            if normalized == prev_norm:
                is_dup = True
                break
            # Near-duplicate check (only if similar length)
            if char_set and prev_set:
                len_ratio = min(len(char_set), len(prev_set)) / max(len(char_set), len(prev_set))
                if len_ratio > 0.7 and _jaccard_similarity(char_set, prev_set) > NEAR_DUPLICATE_THRESHOLD:
                    is_dup = True
                    break

        if is_dup:
            continue

        seen_normalized.append((normalized, char_set))
        result.append(rec)

    return result


# ---------------------------------------------------------------------------
# Step 3: Scene-type filter
# ---------------------------------------------------------------------------

def _filter_scene_types(
    records: list[dict],
    current_state: dict | None = None,
    excluded_types: frozenset[str] | None = None,
) -> list[dict]:
    """Filter out special scene types unless the current state explicitly matches.

    Records with scene_type in excluded_types are removed UNLESS the current
    game state explicitly matches that scene type.

    State matching rules:
    - "birthday" → allowed if state["is_birthday"] is True
    - "festival" → allowed if state["is_festival"] is True
    - "marriage" → allowed if state["relationship"] is spouse/married
    - "gift"     → allowed if state["is_gifting"] is True
    - "special"  → always excluded (WipedMemory/dumped etc. — too complex for state matching)
    """
    if excluded_types is None:
        excluded_types = EXCLUDED_SCENE_TYPES_DEFAULT

    if not excluded_types:
        return records

    state = current_state or {}

    # Determine which special types are allowed by current state
    allowed: set[str] = set()
    if state.get("is_birthday"):
        allowed.add("birthday")
    if state.get("is_festival"):
        allowed.add("festival")
    relationship = (state.get("relationship") or "").lower().strip()
    if relationship in ("spouse", "married"):
        allowed.add("marriage")
    if state.get("is_gifting"):
        allowed.add("gift")
    # "special" is always excluded — no state flag to enable it

    result = []
    for rec in records:
        scene_type = rec.get("scene_type", "daily")
        if scene_type in excluded_types and scene_type not in allowed:
            continue
        result.append(rec)

    return result


# ---------------------------------------------------------------------------
# Step 4: Language priority scoring
# ---------------------------------------------------------------------------

def _lang_score(rec: dict, target_lang: str | None) -> float:
    """Score bonus/penalty based on language match and fallback status.

    Records already annotated with fallback_reason by query_lancedb get
    an additional penalty to rank them below native-language hits.
    """
    if target_lang is None:
        return 0.0
    rec_lang = rec.get("lang", "")
    fallback = rec.get("fallback_reason")

    if rec_lang == target_lang and not fallback:
        return LANG_BONUS
    if rec_lang == "en":
        return 0.0  # neutral — English is acceptable
    if fallback == "fallback_en":
        return -LANG_BONUS
    if fallback == "fallback_any":
        return -LANG_BONUS * 2  # stronger penalty for non-en fallback
    return -LANG_BONUS  # other language → slight penalty


# ---------------------------------------------------------------------------
# Step 5: Relationship closeness scoring
# ---------------------------------------------------------------------------

def _relationship_closeness_score(rec: dict, current_relationship: str) -> float:
    """Score bonus for records whose gate is close to the current relationship.

    Records with a relationship_gate close to the current stage get a bonus.
    The bonus is RELATIONSHIP_CLOSENESS_BONUS * (1 - gap / max_gap).
    A gap of 0 means same stage → full bonus.
    A gap >= max_gap → 0 bonus.

    Records with gate="any" get a penalty when current relationship is friend+,
    because gate=any records often contain stranger-level content (e.g.
    "我不认识你") that is inappropriate at higher relationship levels.
    """
    if not current_relationship:
        return 0.0

    current_stage = _relationship_to_stage(current_relationship)
    if current_stage < 0:
        return 0.0

    gate = rec.get("relationship_gate", "any") or "any"

    # gate=any penalty at friend+ level
    if gate == "any" and current_stage >= 4:
        return -GATE_ANY_PENALTY

    gate_stage = _gate_to_stage(gate)
    if gate_stage < 0:
        return 0.0  # unknown gate → neutral, no bonus

    # Gap: distance between current stage and gate stage
    # For gate <= current (which is guaranteed by the anti-spoiler filter),
    # the closer gate_stage is to current_stage, the better.
    gap = abs(current_stage - gate_stage)
    max_gap = 12  # max possible stage difference (stranger=0 to married=12)

    if gap == 0:
        return RELATIONSHIP_CLOSENESS_BONUS
    elif gap >= max_gap:
        return 0.0
    else:
        return RELATIONSHIP_CLOSENESS_BONUS * (1.0 - gap / max_gap)


# ---------------------------------------------------------------------------
# Step 6: Low-info down-weight
# ---------------------------------------------------------------------------

def _info_score(rec: dict) -> float:
    """Penalise short/generic utterances."""
    text = rec.get("text_display") or rec.get("text_raw") or ""
    if _is_generic(text):
        return -SHORT_PENALTY
    return 0.0


# ---------------------------------------------------------------------------
# Step 7: Scene diversity
# ---------------------------------------------------------------------------

def _enforce_scene_diversity(
    scored: list[tuple[dict, float]],
    max_per_scene: int = MAX_PER_SCENE,
    max_total: int = MAX_OUTPUT,
) -> list[dict]:
    """Greedy pick: at most max_per_scene per scene cluster, up to max_total."""
    scene_counts: dict[str, int] = defaultdict(int)
    selected: list[dict] = []

    for rec, _score in scored:
        if len(selected) >= max_total:
            break

        key = _scene_key(rec)
        if scene_counts[key] >= max_per_scene:
            continue

        selected.append(rec)
        scene_counts[key] += 1

    return selected


# ---------------------------------------------------------------------------
# Step 8: Combined scoring
# ---------------------------------------------------------------------------

def _compute_rank_score(
    rec: dict,
    target_lang: str | None,
    current_relationship: str = "",
) -> float:
    """Compute a ranking score for sort order only.

    Higher is better.  Uses negative distance as the base so that
    lower-distance (more similar) records rank first, then applies
    language bonus, relationship closeness, and low-info penalty.

    NOTE: This is NOT a meaningful "score" — it is only used for
    relative ranking within the candidate set.  Distance filtering
    is done separately via max_distance.
    """
    distance = float(rec.get("_distance", 0.5))
    rank = -distance  # lower distance → higher rank

    # Language adjustment
    rank += _lang_score(rec, target_lang)

    # Relationship closeness
    rank += _relationship_closeness_score(rec, current_relationship)

    # Low-info penalty
    rank += _info_score(rec)

    return rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_examples(
    candidates: list[dict],
    target_lang: str | None = "zh",
    min_output: int = MIN_OUTPUT,
    max_output: int = MAX_OUTPUT,
    max_distance: float | None = None,
    current_state: dict | None = None,
) -> list[dict]:
    """Select 3-5 high-quality few-shot examples from LanceDB candidates.

    Pipeline:
        1. Canonical dedup (keep target_lang variant)
        2. Text dedup (exact + near-duplicate)
        3. Scene-type filter (exclude special events unless matched)
        4. Score each record (vector distance + lang + closeness + info)
        5. Sort by rank score descending
        6. Discard records whose distance > max_distance (if specified)
        7. Enforce scene diversity (max 2 per scene cluster)
        8. Return top 3-5

    Args:
        candidates: Raw results from query_lancedb.search_dialogues().
        target_lang: Preferred output language (e.g. "zh", "en").
        min_output: Minimum number of examples to return.
        max_output: Maximum number of examples to return.
        max_distance: Discard records whose LanceDB distance exceeds this
            threshold.  None means no distance filtering (all candidates
            considered).  Typical LanceDB L2 distances for BGE-M3 range
            0.3–0.8; a value of 0.75 is a reasonable production default.
        current_state: Dict with current game state for filtering/scoring.
            Recognized keys: "relationship", "is_birthday", "is_festival",
            "is_gifting".

    Returns:
        List of 3-5 selected records, each with original LanceDB fields
        plus an optional ``similarity`` field (= 1 − distance).
    """
    if not candidates:
        return []

    state = current_state or {}
    current_relationship = state.get("relationship", "")

    # Step 1: canonical dedup
    deduped = _canonical_dedup(candidates, target_lang)

    # Step 2: text dedup
    deduped = _text_dedup(deduped)

    # Step 3: scene-type filter
    deduped = _filter_scene_types(deduped, state)

    # Step 3.5: gate=any filter at friend+ level
    # gate=any records often contain stranger-level content ("我不认识你")
    # that is inappropriate at friend/close_friend/best_friend/spouse.
    # Remove them when there are ANY alternatives, even if it means fewer examples.
    # (Fewer but appropriate examples > more but inappropriate examples)
    current_stage = _relationship_to_stage(current_relationship)
    if current_stage >= 4:  # friend+
        non_any = [r for r in deduped if (r.get("relationship_gate") or "any") != "any"]
        if non_any:  # only remove gate=any if we have ANY alternatives
            deduped = non_any

    # Step 4: rank
    ranked = [(rec, _compute_rank_score(rec, target_lang, current_relationship)) for rec in deduped]
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Step 5: discard high-distance records
    if max_distance is not None:
        ranked = [
            (rec, rank) for rec, rank in ranked
            if float(rec.get("_distance", 0.5)) <= max_distance
        ]

    # Step 6: scene diversity + cap
    selected = _enforce_scene_diversity(ranked, MAX_PER_SCENE, max_output)

    # Ensure min_output — if not enough, relax scene cap and pull more
    if len(selected) < min_output:
        scene_counts: dict[str, int] = defaultdict(int)
        for rec in selected:
            scene_counts[_scene_key(rec)] += 1

        for rec, _rank in ranked:
            if len(selected) >= min_output:
                break
            if rec in selected:
                continue
            # Allow 1 more per scene for backfill
            key = _scene_key(rec)
            if scene_counts[key] >= MAX_PER_SCENE + 1:
                continue
            selected.append(rec)
            scene_counts[key] += 1

    # Step 7: annotate similarity for downstream consumers
    for rec in selected:
        dist = float(rec.get("_distance", 0.0))
        rec.setdefault("similarity", round(1.0 - dist, 4))

    return selected[:max_output]


def format_examples(records: list[dict]) -> str:
    """Format selected records into the few-shot string for prompt injection.

    Output format:
        - [general_dialogue|heart_min_4] 你在做什么呢？
        - [marriage_dialogue|married] 早上好，亲爱的。 (en)

    Records that came from language fallback show their original lang code
    in parentheses.
    """
    if not records:
        return "No dialogue examples."

    lines = []
    for rec in records:
        dtype = rec.get("dialogue_type", "unknown")
        gate = rec.get("relationship_gate", "unknown")
        text = rec.get("text_display") or rec.get("text_raw") or ""
        fallback = rec.get("fallback_reason")
        if text:
            suffix = f" ({rec.get('lang', '?')})" if fallback else ""
            lines.append(f"- [{dtype}|{gate}] {text}{suffix}")

    return "\n".join(lines) if lines else "No dialogue examples."
