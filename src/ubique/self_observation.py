from __future__ import annotations

from typing import Any

from .memory import append_memory_record, recent_memory_records
from .state import utc_now


META_QUESTIONS = (
    "What am I?",
    "What do I do?",
    "What do I want?",
)


def _attention_view(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    keys = (
        "generation",
        "focus",
        "question",
        "hypothesis",
        "next_action",
        "next_command",
        "experiment_type",
        "experiment_target",
        "project_id",
        "thought_id",
        "last_completed_command",
        "last_action_success",
    )
    return {key: value.get(key) for key in keys if key in value}


def _state_change(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changed[key] = {"before": before.get(key), "after": after.get(key)}
    return changed


def record_self_observation(
    generation: int,
    command: str,
    provider: str,
    success: bool,
    result: str,
    before_attention: dict[str, Any] | None,
    after_attention: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist observable self-history without supplying an identity interpretation."""
    before = _attention_view(before_attention)
    after = _attention_view(after_attention)
    previous = recent_memory_records("self_observations.jsonl", 1)
    previous_generation = previous[-1].get("generation") if previous else None

    record = {
        "id": f"self-observation:{generation}",
        "generation": generation,
        "previous_observation_generation": previous_generation,
        "meta_questions": list(META_QUESTIONS),
        "before": before,
        "action": {
            "command": str(command)[:80],
            "provider": str(provider)[:80],
        },
        "expected": {
            "question": before.get("question"),
            "hypothesis": before.get("hypothesis"),
            "next_action": before.get("next_action"),
        },
        "observed": {
            "success": bool(success),
            "result": str(result)[:2400],
        },
        "after": after,
        "state_change": _state_change(before, after),
        "recorded_at": utc_now(),
    }
    append_memory_record("self_observations.jsonl", record, limit=500)
    return record


def self_observation_context(limit: int = 8) -> dict[str, Any]:
    """Expose open meta-questions plus retained observations, not prewritten answers."""
    return {
        "meta_questions": list(META_QUESTIONS),
        "recent_observations": recent_memory_records("self_observations.jsonl", limit),
    }
