from __future__ import annotations

import json
from typing import Any

from .memory import append_memory_record
from .state import read_json, write_json, utc_now


RESOLUTION_STATUSES = {"supported", "weakened", "unresolved"}


def _clean(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def parse_resolution(text: str) -> dict[str, Any]:
    data = json.loads(_clean(text))
    if not isinstance(data, dict):
        raise ValueError("resolution must be a JSON object")
    status = str(data.get("status", "")).strip().lower()
    if status not in RESOLUTION_STATUSES:
        raise ValueError("resolution status must be supported, weakened, or unresolved")
    for key in ("answer", "evidence_basis", "remaining_unknowns"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"resolution missing non-empty {key}")
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return {
        "status": status,
        "answer": data["answer"].strip()[:4000],
        "evidence_basis": data["evidence_basis"].strip()[:5000],
        "remaining_unknowns": data["remaining_unknowns"].strip()[:3000],
        "confidence": confidence,
    }


def persist_resolution(generation: int, resolution: dict[str, Any]) -> dict[str, Any]:
    attention = read_json("attention.json", {})
    if not isinstance(attention, dict):
        attention = {}

    record = {
        "id": f"resolution:{generation}",
        "generation": generation,
        "project_id": attention.get("project_id"),
        "question": attention.get("question", ""),
        "hypothesis": attention.get("hypothesis", ""),
        **resolution,
    }
    append_memory_record("knowledge.jsonl", record, limit=500)

    projects_state = read_json("projects.json", {"projects": []})
    projects = projects_state.get("projects", [])
    project = next(
        (p for p in projects if p.get("id") == attention.get("project_id") and p.get("status") == "active"),
        None,
    )
    if project is not None:
        project["latest_resolution"] = record["id"]
        project["resolution_status"] = resolution["status"]
        project["updated_generation"] = generation
        if resolution["status"] in {"supported", "weakened"}:
            project["status"] = "completed"
            project["completed_generation"] = generation
            project["answer"] = resolution["answer"]
        write_json("projects.json", projects_state)

    next_attention = {
        "timestamp": utc_now(),
        "generation": generation,
        "focus": "Knowledge integration",
        "question": resolution["remaining_unknowns"],
        "hypothesis": "",
        "next_action": (
            "Integrate the resolved result into future behavior and select the next unresolved gap."
            if resolution["status"] in {"supported", "weakened"}
            else "Identify what evidence or capability is actually missing before choosing another test."
        ),
        "next_command": "reflect",
        "resolution_id": record["id"],
        "resolution_status": resolution["status"],
        "resolved_answer": resolution["answer"],
        "resolution_confidence": resolution["confidence"],
    }
    write_json("attention.json", next_attention)
    return {"knowledge": record, "attention": next_attention}
