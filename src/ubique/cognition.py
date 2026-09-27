from __future__ import annotations

import json
from typing import Any

from .memory import append_memory_record, recent_memory_records
from .state import read_json, write_json, utc_now


ALLOWED_NEXT_COMMANDS = {"reflect", "experiment", "fzg", "evolve"}
ALLOWED_EXPERIMENTS = {"provider_probe", "memory_recall", "memory_abstraction", "hypothesis_ablation", "state_consistency"}


def cognitive_snapshot() -> dict[str, Any]:
    return {
        "attention": read_json("attention.json", {}),
        "projects": read_json("projects.json", {"projects": []}),
        "stagnation": read_json("stagnation.json", {"level": 0, "last_command": None}),
        "recent_thoughts": recent_memory_records("thoughts.jsonl", 5),
        "recent_hypotheses": recent_memory_records("hypotheses.jsonl", 5),
        "recent_concepts": recent_memory_records("concepts.jsonl", 5),
        "recent_project_summaries": recent_memory_records("project_summaries.jsonl", 3),
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
    required = ("observation", "question", "hypothesis", "proposed_experiment", "expected_evidence", "next_action")
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
    experiment_type = str(data.get("experiment_type", "memory_recall")).strip()
    if experiment_type not in ALLOWED_EXPERIMENTS:
        experiment_type = "memory_recall"
    return {
        "observation": data["observation"].strip()[:4000],
        "question": data["question"].strip()[:3000],
        "hypothesis": data["hypothesis"].strip()[:4000],
        "proposed_experiment": data["proposed_experiment"].strip()[:4000],
        "expected_evidence": data["expected_evidence"].strip()[:3000],
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
    hypothesis_id = f"hypothesis:{generation}"
    hypothesis = {
        "id": hypothesis_id,
        "generation": generation,
        "statement": reflection["hypothesis"],
        "proposed_experiment": reflection["proposed_experiment"],
        "expected_evidence": reflection["expected_evidence"],
        "confidence": reflection["confidence"],
        "status": "open",
        "source_thought": thought_id,
    }
    append_memory_record("hypotheses.jsonl", hypothesis, limit=500)
    projects_state = read_json("projects.json", {"projects": []})
    projects = projects_state.setdefault("projects", [])
    title = reflection.get("project_title") or "Autonomous capability development"
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
    attention = {
        "timestamp": utc_now(),
        "generation": generation,
        "focus": active["title"],
        "question": reflection["question"],
        "hypothesis": reflection["hypothesis"],
        "next_action": reflection["next_action"],
        "next_command": reflection["next_command"],
        "experiment_type": reflection.get("experiment_type", "memory_recall"),
        "experiment_target": reflection.get("experiment_target", ""),
        "thought_id": thought_id,
        "project_id": active["id"],
    }
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
    attention["next_command"] = "reflect"
    attention["next_action"] = (
        "Interpret the latest evidence, update the open hypothesis, and choose the next bounded step."
    )
    write_json("attention.json", attention)
    return attention
