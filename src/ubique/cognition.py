from __future__ import annotations

import json
from typing import Any

from .memory import append_memory_record, recent_memory_records
from .state import read_json, write_json, utc_now
from .self_observation import self_observation_context


ALLOWED_NEXT_COMMANDS = {"reflect", "experiment", "fzg", "evolve", "resolve"}
ALLOWED_EXPERIMENTS = {"provider_probe", "memory_recall", "memory_abstraction", "hypothesis_ablation", "state_consistency"}


def _reflection_tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in cleaned.split() if len(token) > 3}


def assess_reflection_stagnation(thoughts: list[dict[str, Any]]) -> dict[str, Any]:
    recent = [
        item for item in thoughts[-8:]
        if isinstance(item, dict) and str(item.get("provisional_answer", "")).strip()
    ]
    if len(recent) < 4:
        return {"detected": False, "reason": "insufficient_history"}

    answers = [str(item.get("provisional_answer", "")) for item in recent]
    questions = [str(item.get("question", "")).strip().lower() for item in recent]
    similarities: list[float] = []
    for left, right in zip(answers, answers[1:]):
        a, b = _reflection_tokens(left), _reflection_tokens(right)
        union = a | b
        similarities.append(len(a & b) / len(union) if union else 1.0)

    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    same_question_count = max((questions.count(q) for q in set(questions) if q), default=0)
    detected = mean_similarity >= 0.62 and same_question_count >= 4
    return {
        "detected": detected,
        "mean_answer_similarity": round(mean_similarity, 4),
        "same_question_count": same_question_count,
        "question": max(set(questions), key=questions.count) if questions else "",
        "recent_thought_ids": [item.get("id") for item in recent],
        "reason": "repeated_provisional_answer" if detected else "sufficient_variation",
    }


def cognitive_snapshot() -> dict[str, Any]:
    recent_thoughts = recent_memory_records("thoughts.jsonl", 8)
    return {
        "attention": read_json("attention.json", {}),
        "projects": read_json("projects.json", {"projects": []}),
        "stagnation": read_json("stagnation.json", {"level": 0, "last_command": None}),
        "recent_thoughts": recent_thoughts,
        "reflection_stagnation": assess_reflection_stagnation(recent_thoughts),
        "recent_hypotheses": recent_memory_records("hypotheses.jsonl", 5),
        "recent_concepts": recent_memory_records("concepts.jsonl", 5),
        "recent_project_summaries": recent_memory_records("project_summaries.jsonl", 3),
        "recent_knowledge": recent_memory_records("knowledge.jsonl", 5),
        "recent_transitions": self_observation_context(3).get("recent_transitions", []),
    }


def _clean_json_text(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def parse_reflection(text: str) -> dict[str, Any]:
    data = json.loads(_clean_json_text(text))
    if not isinstance(data, dict):
        raise ValueError("reflection must be a JSON object")

    required = ("observation", "question", "reflection", "provisional_answer", "uncertainty", "next_action")
    for key in required:
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"reflection missing non-empty {key}")

    next_command = str(data.get("next_command", "reflect")).strip().lower()
    if next_command not in ALLOWED_NEXT_COMMANDS:
        next_command = "reflect"

    def bounded_float(name: str, default: float) -> float:
        try:
            value = float(data.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(0.0, min(1.0, value))

    hypothesis = str(data.get("hypothesis", "")).strip()
    proposed_experiment = str(data.get("proposed_experiment", "")).strip()
    expected_evidence = str(data.get("expected_evidence", "")).strip()
    experiment_type = str(data.get("experiment_type", "")).strip()
    if experiment_type and experiment_type not in ALLOWED_EXPERIMENTS:
        experiment_type = ""

    if next_command == "experiment":
        if not hypothesis:
            raise ValueError("experiment requires a non-empty hypothesis")
        if not proposed_experiment or not expected_evidence:
            raise ValueError("experiment requires proposed_experiment and expected_evidence")
        if not experiment_type:
            raise ValueError("experiment requires a valid experiment_type")

    return {
        "observation": data["observation"].strip()[:4000],
        "question": data["question"].strip()[:3000],
        "reflection": data["reflection"].strip()[:6000],
        "provisional_answer": data["provisional_answer"].strip()[:4000],
        "uncertainty": data["uncertainty"].strip()[:3000],
        "hypothesis": hypothesis[:4000],
        "proposed_experiment": proposed_experiment[:4000],
        "expected_evidence": expected_evidence[:3000],
        "next_action": data["next_action"].strip()[:3000],
        "next_command": next_command,
        "experiment_type": experiment_type,
        "experiment_target": str(data.get("experiment_target", "")).strip()[:80],
        "project_title": str(data.get("project_title", "")).strip()[:200],
        "project_objective": str(data.get("project_objective", "")).strip()[:1200],
        "confidence": bounded_float("confidence", 0.5),
        "importance": bounded_float("importance", 0.5),
    }

def persist_reflection(generation: int, reflection: dict[str, Any]) -> dict[str, Any]:
    thought_id = f"thought:{generation}"
    thought = {"id": thought_id, "generation": generation, **reflection}
    append_memory_record("thoughts.jsonl", thought, limit=500)

    hypothesis_id = None
    project_id = None
    experimental_path = reflection.get("next_command") in {"experiment", "evolve", "fzg"}

    if reflection.get("hypothesis") and experimental_path:
        hypothesis_id = f"hypothesis:{generation}"
        hypothesis = {
            "id": hypothesis_id,
            "generation": generation,
            "statement": reflection["hypothesis"],
            "proposed_experiment": reflection.get("proposed_experiment", ""),
            "expected_evidence": reflection.get("expected_evidence", ""),
            "confidence": reflection["confidence"],
            "status": "open",
            "source_thought": thought_id,
        }
        append_memory_record("hypotheses.jsonl", hypothesis, limit=500)

        projects_state = read_json("projects.json", {"projects": []})
        projects = projects_state.setdefault("projects", [])
        title = reflection.get("project_title") or "Bounded inquiry"
        active = next((p for p in projects if p.get("status") == "active"), None)
        if active is None:
            active = {
                "id": f"project:{generation}",
                "title": title,
                "objective": reflection.get("project_objective") or reflection["question"],
                "status": "active",
                "created_generation": generation,
                "step": 0,
                "history": [],
            }
            projects.append(active)
        active["step"] = int(active.get("step", 0)) + 1
        active["updated_generation"] = generation
        active["latest_thought"] = thought_id
        active["latest_hypothesis"] = hypothesis_id
        history = active.setdefault("history", [])
        history.append({"generation": generation, "thought": thought_id, "next_action": reflection["next_action"]})
        active["history"] = history[-50:]
        write_json("projects.json", projects_state)
        project_id = active["id"]

    attention = {
        "timestamp": utc_now(),
        "generation": generation,
        "focus": reflection.get("project_title") or "Philosophical self-inquiry",
        "question": reflection["question"],
        "reflection": reflection["reflection"],
        "provisional_answer": reflection["provisional_answer"],
        "uncertainty": reflection["uncertainty"],
        "hypothesis": reflection.get("hypothesis", ""),
        "next_action": reflection["next_action"],
        "next_command": reflection["next_command"],
        "thought_id": thought_id,
    }
    if reflection.get("experiment_type"):
        attention["experiment_type"] = reflection["experiment_type"]
    if reflection.get("experiment_target"):
        attention["experiment_target"] = reflection["experiment_target"]
    if hypothesis_id:
        attention["hypothesis_id"] = hypothesis_id
    if project_id:
        attention["project_id"] = project_id
    write_json("attention.json", attention)
    return attention

def update_stagnation(command: str, generation: int) -> dict[str, Any]:
    state = read_json("stagnation.json", {"level": 0, "last_command": None})
    if command in {"status", "fzg"}:
        state["level"] = int(state.get("level", 0)) + 1
    else:
        state["level"] = 0
    state["last_command"] = command
    state["generation"] = generation
    state["timestamp"] = utc_now()
    write_json("stagnation.json", state)
    return state


def record_action_outcome(
    generation: int,
    command: str,
    success: bool,
    result_summary: str,
) -> dict[str, Any]:
    """Return attention to reflection after an experiment/assessment/evolution outcome."""
    attention = read_json("attention.json", {})
    if not isinstance(attention, dict):
        attention = {}

    if command == "experiment":
        append_memory_record(
            "hypotheses.jsonl",
            {
                "id": f"evidence:{generation}",
                "generation": generation,
                "record_type": "evidence",
                "hypothesis": attention.get("hypothesis", ""),
                "experiment_type": attention.get("experiment_type", ""),
                "experiment_target": attention.get("experiment_target", ""),
                "success": success,
                "evidence": result_summary[:2000],
                "status": "observed",
            },
            limit=500,
        )

    attention["timestamp"] = utc_now()
    attention["generation"] = generation
    attention["last_completed_command"] = command
    attention["last_action_success"] = success
    attention["last_evidence"] = result_summary[:2000]
    attention["next_command"] = "resolve"
    attention["next_action"] = (
        "Resolve the current question against the evidence before proposing another experiment or evolution."
    )
    write_json("attention.json", attention)
    return attention
