"""Dialogue trace logger — records per-turn trace data as JSONL.

Each dialogue turn produces one JSON line in logs/dialogue_traces_YYYY-MM-DD.jsonl
with game state, few-shot results, memory retrieval, prompt info, LLM output, and timing.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _PROJECT_ROOT / "logs"
DEBUG_PROMPT_DIR = LOG_DIR / "debug_prompts"

# Set to True to save full prompts to debug_prompts/ (dev only)
SAVE_FULL_PROMPT = os.environ.get("TRACE_SAVE_FULL_PROMPT", "0") == "1"


# ---------------------------------------------------------------------------
# Trace builder — collect data throughout the pipeline, flush once at end
# ---------------------------------------------------------------------------

class DialogueTrace:
    """Collects trace data for a single dialogue turn.

    Usage:
        trace = DialogueTrace()
        trace.set_game_state(state)
        ...
        trace.set_few_shot(result, selected_records)
        trace.set_memory(ctx, raw_results)
        trace.set_prompt_info(prompt, prompt_ctx)
        trace.set_llm_output(raw, final, ...)
        trace.set_timing("few_shot_retrieve", elapsed_ms)
        ...
        trace.flush()
    """

    def __init__(self):
        self._data: dict[str, Any] = {
            "trace_id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timing_ms": {},
        }

    # -- Game state ----------------------------------------------------------

    def set_game_state(self, state: dict) -> None:
        """Extract and record game state fields from the dialogue state dict."""
        self._data.update({
            "npc_id": state.get("npc_id", ""),
            "player_input": state.get("last_user_input", ""),
            "relationship": state.get("relationship", ""),
            "season": state.get("season", ""),
            "day": state.get("dayOfMonth", 0),
            "weather": state.get("weather", ""),
            "location": state.get("location", ""),
            "route_state": state.get("route", "community_center_completed"),
            "game_flags": state.get("game_flags", []),
            "is_birthday": state.get("is_birthday", False),
            "is_festival": state.get("is_festival", False),
            "is_gifting": state.get("is_gifting", False),
        })

    # -- Few-shot ------------------------------------------------------------

    def set_few_shot(self, result, selected_records: list[dict] | None = None) -> None:
        """Record few-shot retrieval result.

        Args:
            result: FewShotResult from get_few_shot_examples()
            selected_records: The actual LanceDB records selected (for detailed trace).
        """
        items = []
        if selected_records:
            for rec in selected_records:
                items.append({
                    "canonical_id": rec.get("canonical_id", ""),
                    "npc_id": rec.get("character", ""),
                    "text_display": rec.get("text_display", "")[:200],
                    "scene_type": rec.get("scene_type", ""),
                    "relationship_gate": rec.get("relationship_gate", ""),
                    "heart_min": rec.get("heart_min", 0),
                    "route": rec.get("route", ""),
                    "required_flags": rec.get("required_flags", ""),
                    "distance": round(float(rec.get("_distance", 0)), 4),
                    "lang": rec.get("lang", ""),
                    "fallback_reason": rec.get("fallback_reason"),
                })
        self._data["few_shot"] = {
            "source": result.source,
            "count": result.count,
            "selected": items,
            "fallback_reason": result.debug.get("fallback_reason"),
            "distance_distribution": result.debug.get("distance_distribution", {}),
        }

    # -- Memory --------------------------------------------------------------

    def set_memory(self, ctx: dict, raw_results: dict | None = None) -> None:
        """Record memory retrieval result.

        Args:
            ctx: The dict returned by ContextManager.get_context() with
                 persona_text, episodic_text, etc.
            raw_results: The raw retrieval dict from MemoryRetriever.retrieve()
                         with per-layer lists of memory dicts.
        """
        items = []
        if raw_results:
            for layer, memories in raw_results.items():
                for m in memories:
                    meta = m.get("metadata", {})
                    items.append({
                        "memory_id": m.get("memory_id", meta.get("memory_id", "")),
                        "memory_type": layer,
                        "content": m.get("content", "")[:200],
                        "importance": meta.get("importance"),
                        "similarity": round(m.get("similarity", 0), 4),
                        "status": meta.get("status", ""),
                    })

        self._data["memory"] = {
            "retrieved_count": len(items),
            "items": items,
        }

    # -- Prompt info ---------------------------------------------------------

    def set_prompt_info(self, prompt_text: str, few_shot_count: int, memory_count: int,
                        has_static_fallback: bool) -> None:
        """Record prompt construction metadata (not the full prompt)."""
        import hashlib
        self._data["prompt"] = {
            "template_version": "dialogue_v1",
            "prompt_chars": len(prompt_text),
            "few_shot_count": few_shot_count,
            "memory_count": memory_count,
            "has_static_fallback": has_static_fallback,
            "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
        }

        if SAVE_FULL_PROMPT:
            DEBUG_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
            prompt_file = DEBUG_PROMPT_DIR / f"{self._data['trace_id']}.txt"
            prompt_file.write_text(prompt_text, encoding="utf-8")

    # -- LLM output ----------------------------------------------------------

    def set_llm_output(
        self,
        raw_output: str,
        final_output: str,
        provider: str = "",
        model: str = "",
        temperature: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Record LLM call result."""
        import re
        forbidden_pattern = re.compile(r"\$[a-z]|#\$b|#\$e|PLAYER_NPC_RELATIONSHIP|%spouse")

        self._data["llm"] = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "raw_output": raw_output[:500] if raw_output else "",
            "final_output": final_output[:500] if final_output else "",
            "output_format_valid": bool(final_output and not error),
            "contains_unknown_markup": bool(raw_output and forbidden_pattern.search(raw_output)),
            "contains_forbidden_token": bool(raw_output and forbidden_pattern.search(raw_output)),
            "error": error,
        }

    # -- Timing --------------------------------------------------------------

    def set_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a pipeline stage."""
        self._data["timing_ms"][stage] = round(elapsed_ms, 1)

    # -- Flush ---------------------------------------------------------------

    def flush(self) -> None:
        """Append the trace as one JSONL line to today's log file."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = LOG_DIR / f"dialogue_traces_{today}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._data, ensure_ascii=False) + "\n")
        except Exception as e:
            # Never let trace logging crash the dialogue pipeline
            print(f"[DialogueTrace] Failed to flush: {e}")


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class StageTimer:
    """Context manager that measures wall-clock time for a pipeline stage.

    Usage:
        with StageTimer(trace, "few_shot_retrieve") as t:
            result = do_work()
    """

    def __init__(self, trace: DialogueTrace, stage: str):
        self.trace = trace
        self.stage = stage
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        self.trace.set_timing(self.stage, elapsed_ms)
