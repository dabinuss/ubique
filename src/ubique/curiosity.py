from __future__ import annotations

from collections import Counter
from typing import Any

from .memory import recent_memory_records
from .state import write_json, utc_now


STANDING_QUESTIONS = ("What am I?", "What do I do?", "What do I want?")


def build_curiosity_snapshot(generation: int) -> dict[str, Any]:
    """Track only standing-question balance without replaying old generated prompts."""
    thoughts = recent_memory_records("thoughts.jsonl", 20)
    questions = [
        str(item.get("question", "")).strip()
        for item in thoughts
        if str(item.get("question", "")).strip()
    ]
    normalized = Counter(q.lower() for q in questions)

    frontiers = [
        {
            "id": "standing-" + str(i + 1),
            "title": q,
            "question": q,
            "recent_mentions": normalized.get(q.lower(), 0),
            "novelty_priority": 1.0 / (1.0 + normalized.get(q.lower(), 0)),
        }
        for i, q in enumerate(STANDING_QUESTIONS)
    ]
    frontiers.sort(key=lambda item: (-item["novelty_priority"], item["question"]))

    result = {
        "timestamp": utc_now(),
        "generation": generation,
        "frontiers": frontiers,
        "standing_questions": frontiers,
        "recent_self_generated_questions": [],
        "rule": (
            "This snapshot only tracks balance among the three standing questions. "
            "Old model-generated question text is intentionally not replayed into future reasoning. "
            "Ubique remains free to generate a new question in the current reflection."
        ),
    }
    write_json("curiosity.json", result)
    return result
