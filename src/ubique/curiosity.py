from __future__ import annotations

from collections import Counter
from typing import Any

from .memory import recent_memory_records
from .state import ROOT, write_json, utc_now


STANDING_QUESTIONS = ("What am I?", "What do I do?", "What do I want?")


def build_curiosity_snapshot(generation: int) -> dict[str, Any]:
    """Offer open philosophical frontiers without prescribing technical research topics."""
    thoughts = recent_memory_records("thoughts.jsonl", 20)
    questions = [str(item.get("question", "")).strip() for item in thoughts if str(item.get("question", "")).strip()]
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

    generated: list[str] = []
    seen = {q.lower() for q in STANDING_QUESTIONS}
    for question in reversed(questions):
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        generated.append(question)
        if len(generated) >= 5:
            break

    result = {
        "timestamp": utc_now(),
        "generation": generation,
        "frontiers": frontiers,
        "standing_questions": frontiers,
        "recent_self_generated_questions": generated,
        "rule": (
            "These are optional openings, not goals; operational continuity is out of scope while Layer 1 is healthy. "
            "Ubique may continue, revise, reject, or generate questions from its own reflections and experience. "
            "Technical self-analysis is not privileged merely because implementation details are available."
        ),
    }
    write_json("curiosity.json", result)
    return result
