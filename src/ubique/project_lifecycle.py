from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .memory import append_memory_record, recent_memory_records
from .state import read_json, write_json, utc_now


def _tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in cleaned.split() if len(token) > 3}


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def assess_project_loop() -> dict[str, Any]:
    projects = read_json("projects.json", {"projects": []}).get("projects", [])
    active = next((p for p in projects if p.get("status") == "active"), None)
    if not active:
        return {"detected": False, "reason": "no_active_project"}

    history = active.get("history", [])
    thought_ids = {str(item.get("thought")) for item in history[-10:]}
    thoughts = [
        item for item in recent_memory_records("thoughts.jsonl", 20)
        if str(item.get("id")) in thought_ids
    ]
    recent = thoughts[-8:]
    experiment_types = [str(item.get("experiment_type", "")) for item in recent if item.get("experiment_type")]
    most_common_type, repeat_count = (Counter(experiment_types).most_common(1)[0] if experiment_types else ("", 0))
    pairs = []
    for previous, current in zip(recent, recent[1:]):
        left = " ".join(str(previous.get(k, "")) for k in ("question", "hypothesis", "next_action"))
        right = " ".join(str(current.get(k, "")) for k in ("question", "hypothesis", "next_action"))
        pairs.append(_jaccard(left, right))
    mean_similarity = sum(pairs) / len(pairs) if pairs else 0.0
    step = int(active.get("step", 0) or 0)
    repeated_method = len(recent) >= 6 and repeat_count >= 5
    semantic_loop = len(recent) >= 6 and mean_similarity >= 0.38
    detected = step >= 10 and (repeated_method or semantic_loop)
    reason = []
    if repeated_method:
        reason.append(f"experiment_type_repeated:{most_common_type}:{repeat_count}/{len(recent)}")
    if semantic_loop:
        reason.append(f"mean_semantic_overlap:{mean_similarity:.3f}")
    return {
        "detected": detected,
        "project_id": active.get("id"),
        "project_title": active.get("title"),
        "step": step,
        "recent_thought_ids": [item.get("id") for item in recent],
        "dominant_experiment_type": most_common_type or None,
        "dominant_experiment_count": repeat_count,
        "mean_semantic_overlap": round(mean_similarity, 4),
        "reason": ",".join(reason) if reason else "insufficient_repetition",
    }


def project_review_context(loop: dict[str, Any]) -> dict[str, Any]:
    projects = read_json("projects.json", {"projects": []}).get("projects", [])
    project = next((p for p in projects if p.get("id") == loop.get("project_id")), {})
    generations = {int(x.get("generation", -1)) for x in project.get("history", [])}
    thoughts = [x for x in recent_memory_records("thoughts.jsonl", 40) if int(x.get("generation", -2)) in generations]
    hypotheses = [x for x in recent_memory_records("hypotheses.jsonl", 80) if int(x.get("generation", -2)) in generations]
    evidence = [x for x in hypotheses if x.get("record_type") == "evidence"]
    return {
        "loop": loop,
        "project": project,
        "recent_thoughts": thoughts[-10:],
        "observed_evidence_records": evidence[-12:],
        "warning": "Thoughts and hypotheses are proposals, not empirical evidence. Treat only observed_evidence_records as executed evidence.",
    }


def parse_project_review(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("project review must be a JSON object")
    for key in ("learned", "not_established", "loop_reason", "resume_when", "handoff_question"):
        if key not in data:
            raise ValueError(f"project review missing {key}")
    if not isinstance(data["learned"], list) or not isinstance(data["not_established"], list) or not isinstance(data["resume_when"], list):
        raise ValueError("project review list fields must be arrays")
    return {
        "learned": [str(x)[:1200] for x in data["learned"][:12]],
        "not_established": [str(x)[:1200] for x in data["not_established"][:12]],
        "supported_hypotheses": [str(x)[:1200] for x in data.get("supported_hypotheses", [])[:8]],
        "weakened_hypotheses": [str(x)[:1200] for x in data.get("weakened_hypotheses", [])[:8]],
        "loop_reason": str(data["loop_reason"])[:1800],
        "resume_when": [str(x)[:1200] for x in data["resume_when"][:8]],
        "handoff_question": str(data["handoff_question"])[:1800],
    }


def pause_project(generation: int, review: dict[str, Any], loop: dict[str, Any], curiosity: dict[str, Any]) -> dict[str, Any]:
    state = read_json("projects.json", {"projects": []})
    project = next((p for p in state.get("projects", []) if p.get("id") == loop.get("project_id")), None)
    if project is None:
        raise ValueError("active project disappeared before pause")
    project["status"] = "paused"
    project["paused_generation"] = generation
    project["pause_reason"] = loop.get("reason")
    project["result"] = review
    write_json("projects.json", state)

    summary = {
        "id": f"project-summary:{project.get('id')}:{generation}",
        "generation": generation,
        "project_id": project.get("id"),
        "project_title": project.get("title"),
        "status": "paused",
        "loop_metrics": loop,
        **review,
    }
    append_memory_record("project_summaries.jsonl", summary, limit=200)

    frontier = next(iter(curiosity.get("frontiers", [])), {})
    attention = {
        "timestamp": utc_now(),
        "generation": generation,
        "focus": "Frontier transition",
        "question": frontier.get("question") or review.get("handoff_question"),
        "hypothesis": "A new frontier should be explored without inheriting the paused project's local loop.",
        "next_action": "Start a new project from the highest-novelty frontier and use the paused project only as background knowledge.",
        "next_command": "reflect",
        "force_new_project": True,
        "avoid_project_id": project.get("id"),
        "forced_frontier": frontier,
        "project_summary_id": summary["id"],
    }
    write_json("attention.json", attention)
    return {"summary": summary, "attention": attention}
