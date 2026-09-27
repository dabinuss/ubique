from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory import recent_memory_records
from .state import ROOT, read_json, write_json, utc_now


FRONTIERS = [
    {
        "id": "hypothesis_revision",
        "title": "Hypothesis revision",
        "question": "How should earlier hypotheses change when new evidence conflicts with them?",
        "keywords": ["hypothesis", "evidence", "revision", "conflict"],
    },
    {
        "id": "memory_abstraction",
        "title": "Memory abstraction",
        "question": "Can repeated episodes be compressed into reusable concepts without losing important exceptions?",
        "keywords": ["memory", "concept", "abstraction", "episode"],
    },
    {
        "id": "multi_step_planning",
        "title": "Multi-step planning",
        "question": "Can an autonomous project preserve a coherent objective across several generations and adapt its plan from evidence?",
        "keywords": ["project", "plan", "step", "objective"],
    },
    {
        "id": "experiment_design",
        "title": "Experiment design",
        "question": "Can Ubique distinguish a useful falsifiable experiment from an action that merely produces more telemetry?",
        "keywords": ["experiment", "falsifiable", "control", "evidence"],
    },
    {
        "id": "capability_composition",
        "title": "Capability composition",
        "question": "Which existing capabilities can be combined into a genuinely new solution path rather than exercised independently?",
        "keywords": ["capability", "combine", "compose", "solution"],
    },
    {
        "id": "question_generation",
        "title": "Question generation",
        "question": "What makes an internally generated question worth pursuing when no external task requires it?",
        "keywords": ["question", "curiosity", "worth", "pursue"],
    },
]


def _text(records: list[dict[str, Any]]) -> str:
    return " ".join(str(value).lower() for record in records for value in record.values())


def build_curiosity_snapshot(generation: int) -> dict[str, Any]:
    """Build Layer-2 frontiers without feeding operational health back into attention."""
    thoughts = recent_memory_records("thoughts.jsonl", 12)
    hypotheses = recent_memory_records("hypotheses.jsonl", 12)
    corpus = _text(thoughts + hypotheses)
    scored = []
    for frontier in FRONTIERS:
        mentions = sum(corpus.count(keyword) for keyword in frontier["keywords"])
        scored.append({
            "id": frontier["id"],
            "title": frontier["title"],
            "question": frontier["question"],
            "prior_mentions": mentions,
            "novelty_priority": 1.0 / (1.0 + mentions),
        })
    scored.sort(key=lambda item: (-item["novelty_priority"], item["id"]))

    source_modules = sorted(
        p.stem for p in (ROOT / "src" / "ubique").glob("*.py")
        if p.stem not in {"__init__", "__main__"}
    )
    result = {
        "timestamp": utc_now(),
        "generation": generation,
        "frontiers": scored[:4],
        "source_modules": source_modules,
        "rule": "When Layer 1 is healthy, choose epistemic/capability questions; operational continuity is out of scope unless new evidence makes it relevant.",
    }
    write_json("curiosity.json", result)
    return result
