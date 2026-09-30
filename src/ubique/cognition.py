from __future__ import annotations

import json
from typing import Any

from .memory import append_memory_record, recent_memory_records, recent_episodes
from .state import read_json, write_json, utc_now

ALLOWED_NEXT_COMMANDS = {"reflect", "experiment", "evolve", "resolve", "library"}
ALLOWED_EXPERIMENTS = {"provider_probe", "memory_recall", "memory_abstraction", "hypothesis_ablation", "state_consistency"}
ALLOWED_LIBRARY_ACTIONS = {"list", "read", "add", "request", "note"}


def _tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {t for t in cleaned.split() if len(t) > 3}


def _similarity(left_text: str, right_text: str) -> float:
    left, right = _tokens(left_text), _tokens(right_text)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def assess_reflection_stagnation(thoughts: list[dict[str, Any]]) -> dict[str, Any]:
    recent = [
        x for x in thoughts[-8:]
        if isinstance(x, dict) and str(x.get("provisional_answer", "")).strip()
    ]
    if len(recent) < 2:
        return {"detected": False, "reason": "insufficient_history"}

    last_two = recent[-2:]
    pair_similarity = _similarity(
        str(last_two[0].get("provisional_answer", "")),
        str(last_two[1].get("provisional_answer", "")),
    )
    last_questions = [str(x.get("question", "")).strip().lower() for x in last_two]
    consecutive_repeat = (
        bool(last_questions[0])
        and last_questions[0] == last_questions[1]
        and pair_similarity >= 0.50
    )

    answers = [str(x.get("provisional_answer", "")) for x in recent]
    questions = [str(x.get("question", "")).strip().lower() for x in recent]
    sims = [_similarity(a, b) for a, b in zip(answers, answers[1:])]
    mean_similarity = sum(sims) / len(sims) if sims else 0.0
    same_question_count = max((questions.count(q) for q in set(questions) if q), default=0)
    longer_loop = len(recent) >= 4 and mean_similarity >= 0.45 and same_question_count >= 3
    detected = consecutive_repeat or longer_loop

    repeated_question = last_questions[-1] if consecutive_repeat else (
        max(set(questions), key=questions.count) if questions else ""
    )
    return {
        "detected": detected,
        "pair_similarity": round(pair_similarity, 4),
        "mean_answer_similarity": round(mean_similarity, 4),
        "same_question_count": same_question_count,
        "question": repeated_question,
        "recent_thought_ids": [x.get("id") for x in recent],
        "reason": (
            "repeated_consecutive_answer"
            if consecutive_repeat
            else "repeated_provisional_answer"
            if longer_loop
            else "sufficient_variation"
        ),
    }


def _safe_library_fact(result_text: str) -> dict[str, Any]:
    try:
        data = json.loads(result_text)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    item = data.get("item") if isinstance(data.get("item"), dict) else {}
    return {
        "action": str(data.get("action", ""))[:80],
        "item_id": str(item.get("id") or data.get("item_id") or "")[:120],
        "title": str(item.get("title", ""))[:300],
        "content_available": bool(data.get("content_available", item.get("content_path"))),
    }


def _executed_action_facts() -> list[dict[str, Any]]:
    """Return system-recorded actions without replaying model-authored reflection prose."""
    out: list[dict[str, Any]] = []
    for item in recent_episodes(30):
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "")).strip()
        if not command:
            continue

        fact: dict[str, Any] = {
            "generation": item.get("generation"),
            "command": command,
            "success": bool(item.get("success")),
        }
        if item.get("deferred"):
            fact["deferred"] = True

        if command in {"experiment", "evolve", "resolve"}:
            fact["recorded_result"] = str(item.get("result", ""))[:1800]
        elif command == "library":
            fact["library_action"] = _safe_library_fact(str(item.get("result", "")))
        # reflect/think/plan are recorded only as actions. Their prose is interpretation,
        # not evidence and must not return through the observation channel.

        out.append(fact)
    return out[-12:]


def deterministic_observation_summary(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return "No recent executed-action facts are recorded."
    parts: list[str] = []
    for fact in facts[-6:]:
        generation = fact.get("generation")
        command = str(fact.get("command", "action"))
        if fact.get("deferred"):
            status = "deferred"
        else:
            status = "succeeded" if fact.get("success") else "failed"
        parts.append(f"generation {generation}: {command} {status}")
    return "Recorded executed actions: " + "; ".join(parts) + "."


def cognitive_snapshot() -> dict[str, Any]:
    thoughts = recent_memory_records("thoughts.jsonl", 8)
    return {
        "attention": read_json("attention.json", {}),
        "projects": read_json("projects.json", {"projects": []}),
        "stagnation": read_json("stagnation.json", {"level": 0, "last_command": None}),
        "recent_thoughts": thoughts,
        "reflection_stagnation": assess_reflection_stagnation(thoughts),
        "recent_hypotheses": recent_memory_records("hypotheses.jsonl", 5),
        "recent_concepts": recent_memory_records("concepts.jsonl", 5),
        "recent_project_summaries": recent_memory_records("project_summaries.jsonl", 3),
        "recent_knowledge": recent_memory_records("knowledge.jsonl", 5),
        "executed_action_facts": _executed_action_facts(),
    }


def _clean_json_text(text: str) -> str:
    value = text.strip()
    fence = chr(96) * 3
    if value.startswith(fence):
        lines = value.splitlines()
        if lines and lines[0].startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def parse_reflection(text: str) -> dict[str, Any]:
    data = json.loads(_clean_json_text(text))
    if not isinstance(data, dict):
        raise ValueError("reflection must be a JSON object")

    interpretation = data.get("interpretation")
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ValueError("reflection missing non-empty interpretation")

    for key in ("question", "provisional_answer", "uncertainty", "next_action"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"reflection missing non-empty {key}")
    raw_assumptions = data.get("assumptions", [])
    if raw_assumptions is None:
        raw_assumptions = []
    if not isinstance(raw_assumptions, list):
        raise ValueError("reflection assumptions must be a list")
    assumptions = [
        str(item).strip()[:1200]
        for item in raw_assumptions[:12]
        if isinstance(item, str) and str(item).strip()
    ]

    technical_markers = (
        "architecture",
        "training",
        "token window",
        "token context",
        "memory mechanism",
        "internal state",
        "sensor",
        "affect",
        "persistent substrate",
        "language model",
        "model weights",
    )
    epistemic_text = " ".join(
        [
            interpretation,
            str(data.get("provisional_answer", "")),
            str(data.get("next_action", "")),
        ]
    ).lower()
    if any(marker in epistemic_text for marker in technical_markers) and not assumptions:
        raise ValueError(
            "technical interpretation requires explicit assumptions instead of unsupported factual claims"
        )

    next_command = str(data.get("next_command", "reflect")).strip().lower()
    if next_command not in ALLOWED_NEXT_COMMANDS:
        next_command = "reflect"

    hypothesis = str(data.get("hypothesis", "")).strip()
    proposed = str(data.get("proposed_experiment", "")).strip()
    expected = str(data.get("expected_evidence", "")).strip()
    experiment_type = str(data.get("experiment_type", "")).strip()
    experiment_target = str(data.get("experiment_target", "")).strip()[:80]
    if experiment_type and experiment_type not in ALLOWED_EXPERIMENTS:
        experiment_type = ""

    planning_note = ""
    if next_command == "experiment":
        missing: list[str] = []
        if not hypothesis:
            missing.append("hypothesis")
        if not proposed:
            missing.append("proposed_experiment")
        if not expected:
            missing.append("expected_evidence")
        if not experiment_type:
            missing.append("valid experiment_type")
        if missing:
            next_command = "reflect"
            planning_note = (
                "Incomplete experiment request was kept as reflection instead of failing: "
                + ", ".join(missing)
            )

    library_action = str(data.get("library_action", "")).strip().lower()
    library_item_id = str(data.get("library_item_id", "")).strip()[:120]
    library_title = str(data.get("library_title", "")).strip()[:300]
    library_text = str(data.get("library_text", ""))[:200000]
    library_source = str(data.get("library_source", "")).strip()[:1000]
    library_reason = str(data.get("library_reason", "")).strip()[:1500]
    if next_command == "library":
        bad = library_action not in ALLOWED_LIBRARY_ACTIONS
        bad = bad or (library_action in {"read", "note"} and not library_item_id)
        bad = bad or (library_action in {"add", "request"} and not library_title)
        bad = bad or (library_action == "note" and not library_text.strip())
        if bad:
            next_command = "reflect"
            planning_note = "Invalid library request was kept as reflection instead of failing."

    def bounded(name: str, default: float) -> float:
        try:
            value = float(data.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(0.0, min(1.0, value))

    return {
        "question": data["question"].strip()[:3000],
        "reflection": interpretation.strip()[:6000],
        "interpretation_status": "model_interpretation_not_evidence",
        "assumptions": assumptions,
        "provisional_answer": data["provisional_answer"].strip()[:4000],
        "uncertainty": data["uncertainty"].strip()[:3000],
        "hypothesis": hypothesis[:4000],
        "proposed_experiment": proposed[:4000],
        "expected_evidence": expected[:3000],
        "next_action": data["next_action"].strip()[:3000],
        "next_command": next_command,
        "experiment_type": experiment_type,
        "experiment_target": experiment_target,
        "library_action": library_action,
        "library_item_id": library_item_id,
        "library_title": library_title,
        "library_text": library_text,
        "library_source": library_source,
        "library_reason": library_reason,
        "planning_note": planning_note,
        "confidence": bounded("confidence", 0.5),
        "importance": bounded("importance", 0.5),
    }


def persist_reflection(
    generation: int,
    reflection: dict[str, Any],
    observed_facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    facts = list(observed_facts or [])
    observation = deterministic_observation_summary(facts)
    thought_id = f"thought:{generation}"
    thought = {
        "id": thought_id,
        "generation": generation,
        "observation": observation,
        "observation_status": "recorded_system_facts",
        "observed_facts": facts,
        **reflection,
    }
    append_memory_record("thoughts.jsonl", thought, limit=500)

    hypothesis_id = None
    project_id = None
    if reflection.get("hypothesis") and reflection.get("next_command") in {"experiment", "evolve"}:
        hypothesis_id = f"hypothesis:{generation}"
        append_memory_record(
            "hypotheses.jsonl",
            {
                "id": hypothesis_id,
                "generation": generation,
                "statement": reflection["hypothesis"],
                "proposed_experiment": reflection.get("proposed_experiment", ""),
                "expected_evidence": reflection.get("expected_evidence", ""),
                "confidence": reflection["confidence"],
                "status": "open",
                "source_thought": thought_id,
            },
            limit=500,
        )
        projects_state = read_json("projects.json", {"projects": []})
        projects = projects_state.setdefault("projects", [])
        active = next((p for p in projects if p.get("status") == "active"), None)
        if active is None:
            active = {
                "id": f"project:{generation}",
                "title": "Bounded inquiry",
                "objective": reflection["question"],
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
        active.setdefault("history", []).append(
            {
                "generation": generation,
                "thought": thought_id,
                "next_action": reflection["next_action"],
            }
        )
        active["history"] = active["history"][-50:]
        write_json("projects.json", projects_state)
        project_id = active["id"]

    attention = {
        "timestamp": utc_now(),
        "generation": generation,
        "focus": "Philosophical self-inquiry",
        "observation": observation,
        "observation_status": "recorded_system_facts",
        "observed_facts": facts,
        "question": reflection["question"],
        "reflection": reflection["reflection"],
        "interpretation_status": "model_interpretation_not_evidence",
        "assumptions": list(reflection.get("assumptions", [])),
        "provisional_answer": reflection["provisional_answer"],
        "uncertainty": reflection["uncertainty"],
        "hypothesis": reflection.get("hypothesis", ""),
        "next_action": reflection["next_action"],
        "next_command": reflection["next_command"],
        "thought_id": thought_id,
    }
    if reflection.get("planning_note"):
        attention["planning_note"] = reflection["planning_note"]
    if reflection.get("experiment_type"):
        attention["experiment_type"] = reflection["experiment_type"]
    if reflection.get("experiment_target"):
        attention["experiment_target"] = reflection["experiment_target"]
    if reflection.get("next_command") == "library":
        for key in (
            "library_action",
            "library_item_id",
            "library_title",
            "library_text",
            "library_source",
            "library_reason",
        ):
            if reflection.get(key):
                attention[key] = reflection[key]
    if hypothesis_id:
        attention["hypothesis_id"] = hypothesis_id
    if project_id:
        attention["project_id"] = project_id
    write_json("attention.json", attention)
    return attention


def record_reflection_deferred(generation: int, reason: str) -> dict[str, Any]:
    """Record a clean pause when no remote model can produce a reflection."""
    attention = read_json("attention.json", {})
    if not isinstance(attention, dict):
        attention = {}
    attention.update(
        {
            "timestamp": utc_now(),
            "generation": generation,
            "focus": "Philosophical self-inquiry",
            "reflection_deferred": True,
            "deferred_reason": reason[:1200],
            "next_command": "reflect",
            "next_action": "Retry philosophical reflection when a remote reasoning provider is available.",
        }
    )
    write_json("attention.json", attention)
    return attention


def update_stagnation(command: str, generation: int) -> dict[str, Any]:
    state = read_json("stagnation.json", {"level": 0, "last_command": None})
    state["level"] = int(state.get("level", 0)) + 1 if command == "status" else 0
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
    attention.update(
        {
            "timestamp": utc_now(),
            "generation": generation,
            "last_completed_command": command,
            "last_action_success": success,
            "last_evidence": result_summary[:2000],
            "next_command": "resolve",
            "next_action": (
                "Resolve the current question against the executed outcome before "
                "proposing another experiment or evolution."
            ),
        }
    )
    write_json("attention.json", attention)
    return attention


def record_library_outcome(generation: int, result: dict[str, Any]) -> dict[str, Any]:
    attention = read_json("attention.json", {})
    if not isinstance(attention, dict):
        attention = {}
    attention.update(
        {
            "timestamp": utc_now(),
            "generation": generation,
            "last_completed_command": "library",
            "library_context": result,
            "next_command": "reflect",
            "next_action": (
                "Reflect on whether the voluntarily selected library material or request "
                "changes, enriches, or leaves unchanged the current view. "
                "Do not treat library text as executed evidence."
            ),
        }
    )
    for key in (
        "library_action",
        "library_item_id",
        "library_title",
        "library_text",
        "library_source",
        "library_reason",
    ):
        attention.pop(key, None)
    write_json("attention.json", attention)
    return attention
