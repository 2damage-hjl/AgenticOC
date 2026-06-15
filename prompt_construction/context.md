# context.md

## 1. Project Goal

This project builds a Stardew Valley dialogue agent with retrieval-augmented few-shot prompting.

The current milestone is to run the following pipeline end-to-end:

```text
data/game_scripts
  -> canonical dialogue records
  -> processed multilingual records
  -> LanceDB multilingual vector index
  -> query-time retrieval with game-state gates
```

The first scope is limited to:

```text
Dialogue + Marriage + key parser + required_flags + multilingual retrieval
```

Do not expand to Festivals or Strings until this milestone is stable.

---

## 2. Core Principles

### 2.1 Source of truth

`data/game_scripts/` contains unpacked original game assets and is the authoritative source.

Rules:

```text
Do not edit files under data/game_scripts/
Do not manually add annotations inside game_scripts
Do not overwrite unpacked original files
```

Derived data must be written to:

```text
data/canonical/
data/processed/
data/indexes/
```

### 2.2 Deterministic parsing before LLM annotation

Game script keys often contain hard conditions.

For example:

```text
winter_Wed2 -> season = winter, day_of_week = Wed, heart_min = 2
```

These conditions must be parsed by code, not guessed by an LLM.

LLM annotation is allowed only for semantic fields such as:

```text
topic_tags
intent
tone
attitude
length_bucket
```

LLM annotation must not override deterministic script conditions.

### 2.3 Canonical records first

The first extraction output must be canonical records.

Canonical records should preserve original author keys, source files, character names, multilingual text, and script-derived conditions.

Canonical records should not contain embeddings.

### 2.4 Multilingual design

Do not create separate logical databases for each language.

Use one canonical dialogue record with multiple localized texts:

```json
"texts": {
  "en": "...",
  "zh": "...",
  "ja": "..."
}
```

When building the vector index, expand each canonical record into one vector row per available language:

```text
canonical_id + lang = one vector record
```

At query time:

```text
Prefer target_lang
Fallback to en
Fallback to any available language only if necessary
```

### 2.5 Data boundary

Do not rely on the base LLM's memory for game-state constraints.

The model may know Stardew Valley, but the system must control:

```text
relationship gates
heart gates
marriage status
route state
location unlocks
festival/date constraints
spoiler-prone unlocks
```

Game-state constraints must be handled by metadata gates, not by prompt instructions alone.

---

## 3. Directory Convention

Use the following structure:

```text
project_root/
├── data/
│   ├── game_scripts/
│   │   ├── Dialogue/
│   │   ├── Festivals/
│   │   └── Strings/
│   ├── canonical/
│   │   └── dialogue/
│   ├── processed/
│   └── indexes/
│       └── lancedb/
│
├── configs/
│   ├── dialogue_key_rules.yaml
│   ├── dialogue_gates.yaml
│   └── index_config.yaml
│
├── scripts/
│   ├── extract_dialogue_canonical.py
│   ├── build_processed_dialogues.py
│   └── build_lancedb_index.py
│
├── retrieval/
│   ├── query_lancedb.py
│   └── example_selector.py
│
├── utils/
│   ├── json_io.py
│   ├── text_normalize.py
│   ├── dialogue_key_parser.py
│   ├── dialogue_token_parser.py
│   └── hashing.py
│
└── tests/
    ├── test_key_parser.py
    ├── test_canonical_extraction.py
    ├── test_required_flags.py
    └── test_multilingual_retrieval.py
```

---

## 4. Canonical Schema

Canonical records are produced from `data/game_scripts/Dialogue` and marriage dialogue assets.

Minimum schema:

```json
{
  "canonical_id": "Dialogue/Abigail:winter_Wed2",
  "author_key": "winter_Wed2",
  "character": "Abigail",
  "dialogue_type": "general_dialogue",
  "texts": {
    "en": "...",
    "zh": "..."
  },
  "source": {
    "source_type": "verified_game_asset",
    "collection": "Dialogue",
    "subcollection": "Characters",
    "files": {
      "en": "data/game_scripts/Dialogue/en/Abigail.json",
      "zh": "data/game_scripts/Dialogue/zh/Abigail.json"
    }
  },
  "script_conditions": {
    "season": ["winter"],
    "day_of_week": ["Wed"],
    "day_of_month": ["any"],
    "heart_min": 2,
    "relationship_gate": "heart_min_2",
    "weather": ["any"],
    "scene": "any",
    "special_key_type": "generic_dialogue",
    "parse_warnings": []
  },
  "control": {
    "required_flags": [],
    "route": "any"
  },
  "quality": {
    "source_confidence": "high",
    "script_parse_confidence": "high"
  }
}
```

### 4.1 Required field naming

Use `required_flags`, not `required`.

Reason:

```text
required is a JSON Schema keyword;
required_flags is the project-specific game-state gate field.
```

### 4.2 dialogue_type enum

Allowed values for current milestone:

```text
general_dialogue
marriage_dialogue
```

Future values are allowed only after this milestone is stable:

```text
festival_dialogue
string_dialogue
event_dialogue
gift_reaction
movie_dialogue
```

### 4.3 control.route enum

Allowed values:

```text
any
pre_choice
non_joja
community_center
joja
post_joja
```

### 4.4 control.required_flags

`required_flags` must be an array.

If no flags are needed, use:

```json
"required_flags": []
```

Do not use `null`.
Do not omit the field in processed data.
Do not let an LLM invent new flags.

---

## 5. Marriage Dialogue Rules

Marriage dialogue may be stored separately in the original game assets.

Extraction may produce separate intermediate files:

```text
data/canonical/dialogue/Abigail.general.json
data/canonical/dialogue/Abigail.marriage.json
```

However, processed data and indexes must use one unified dialogue pool.

For marriage dialogue:

```text
dialogue_type = marriage_dialogue
relationship_gate = married_to_character
required_flags = ["relationship.married_to.{Character}"]
route = any
```

Example:

```json
"control": {
  "required_flags": ["relationship.married_to.Abigail"],
  "route": "any"
}
```

Do not ask the LLM whether a marriage file line is married dialogue. It is deterministic from source structure.

---

## 6. Dialogue Key Parsing Rules

The key parser must parse high-confidence rules first.

Examples:

```text
winter_Wed2
  -> season = winter
  -> day_of_week = Wed
  -> heart_min = 2

summer_Mon8
  -> season = summer
  -> day_of_week = Mon
  -> heart_min = 8

spring_13
  -> season = spring
  -> day_of_month = 13

rain
  -> weather = rainy

GreenRain
  -> weather = green_rain

Resort / Resort_* 
  -> scene = island_resort
  -> required_flags includes world.ginger_island.beach_resort.opened
```

The parser should produce parse warnings when it detects partial patterns but cannot fully parse the key.

LLM output must never override key parser output.

---

## 7. Text Token Handling

Original Stardew dialogue contains special tokens and commands.

Maintain three text forms in processed records:

```text
raw: original game script text
prompt: cleaned text for few-shot prompt
embedding: cleaned and normalized text for vector embedding
```

Rules:

```text
raw must preserve original text
prompt may replace @ with {player_name}
embedding may replace @ with player
embedding should remove or normalize dialogue control tokens
```

Do not over-clean raw text.

Common tokens to handle:

```text
@          player name placeholder
#$b#       line break / paragraph break
$h $s etc portrait/emotion commands
```

Token parsing should be deterministic and implemented in `utils/dialogue_token_parser.py`.

---

## 8. Processed Schema

Processed records are language-expanded.

One processed record corresponds to:

```text
canonical_id + lang
```

Minimum schema:

```json
{
  "canonical_id": "Dialogue/Abigail:winter_Wed2",
  "record_id": "Dialogue/Abigail:winter_Wed2:zh",
  "character": "Abigail",
  "lang": "zh",
  "dialogue_type": "general_dialogue",
  "text": {
    "raw": "...",
    "prompt": "...",
    "embedding": "..."
  },
  "context": {
    "author_key": "winter_Wed2",
    "season": ["winter"],
    "day_of_week": ["Wed"],
    "day_of_month": ["any"],
    "heart_min": 2,
    "relationship_gate": "heart_min_2",
    "weather": ["any"],
    "scene": "any"
  },
  "control": {
    "required_flags": [],
    "route": "any"
  },
  "quality": {
    "source_confidence": "high",
    "script_parse_confidence": "high",
    "curation_status": "unannotated"
  },
  "indexing": {
    "text_hash": "...",
    "retrieval_text_hash": "...",
    "embedding_model": "BAAI/bge-m3",
    "vector_ref": "Dialogue/Abigail:winter_Wed2:zh"
  }
}
```

---

## 9. Vector Index Rules

Use LanceDB for the current milestone.

Vector table path:

```text
data/indexes/lancedb/
```

Each row should include:

```text
record_id
canonical_id
character
lang
text_prompt
text_embedding
vector
dialogue_type
season
day_of_week
heart_min
relationship_gate
weather
scene
required_flags
route
source_confidence
script_parse_confidence
```

At query time:

```text
1. Filter by character
2. Filter by lang = target_lang
3. Apply game-state gates: required_flags and route
4. Search vector index
5. If result count is too low, fallback to lang = en
6. If still too low, fallback to any available language
```

Do not directly use vector top-k as final few-shot selection.
Vector search returns candidates only.
Final selection happens in `retrieval/example_selector.py`.

---

## 10. Code Style Rules

### 10.1 General

Use Python 3.10+.

Use type hints for public functions.

Prefer pure functions in `utils/`.

Scripts should be thin orchestration layers.

### 10.2 Path handling

Use `pathlib.Path`.

Do not hard-code absolute paths.

All paths should be relative to project root or read from config.

### 10.3 JSON handling

Use UTF-8.

Use `ensure_ascii=False`.

Use stable indentation:

```python
json.dump(data, f, ensure_ascii=False, indent=2)
```

### 10.4 Error handling

Do not silently drop records.

When skipping a record, write a warning or add a parse warning.

Extraction scripts should produce a small report:

```text
total_files
total_records
records_with_parse_warnings
missing_language_counts
skipped_files
```

### 10.5 Determinism

Extraction must be deterministic.

Sort files before processing.

Sort output records by `canonical_id`.

Sort language keys in output where possible.

### 10.6 No hidden mutation

`game_scripts` is read-only.

Generated outputs go only to canonical, processed, or indexes.

---

## 11. Testing Rules

Every parser or extractor must have tests.

Minimum tests:

```text
winter_Wed2 parses correctly
summer_Mon8 parses correctly
rain key maps to rainy weather
GreenRain maps to green_rain
Resort key adds island resort required flag
marriage source adds relationship.married_to.Character
multilingual texts align under one canonical_id
processed records expand by language
LanceDB query returns target language first
```

Tests should verify both positive and boundary cases.

---

## 12. Out of Scope for Current Milestone

Do not implement these yet unless explicitly requested:

```text
Festivals extraction
Strings extraction
full LLM semantic annotation
MMR selector
duplicate grouping
incremental embedding cache
all-NPC production indexing
human review UI
```

They can be added after the current milestone passes tests.

---

## 13. Final Reminder

The current goal is not to build a perfect Stardew encyclopedia.

The current goal is to build a reliable data foundation:

```text
Dialogue + Marriage + key parser + required_flags + multilingual retrieval
```
