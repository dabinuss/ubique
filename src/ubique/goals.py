from __future__ import annotations

import json
from typing import Any

from .models import Task


def _compact(value: Any, limit: int = 7000) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:limit]


def choose_endogenous_goal(
    generation: int,
    homeostasis: dict[str, Any],
    remote_reasoning_available: bool,
    telemetry: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    cognition: dict[str, Any] | None = None,
) -> Task:
    """Layer 1 guards operation; layer 2 continuously develops when healthy."""
    needs = homeostasis.get("needs", [])
    telemetry = telemetry or {}
    environment = environment or {}
    preflight = preflight or {"development_allowed": True}
    cognition = cognition or {}
    measured_context = {
        "homeostasis": homeostasis,
        "fzg_telemetry": telemetry,
        "environment": environment,
        "preflight": preflight,
    }
    critical = [n for n in needs if n.get("level") == "critical"]

    if critical or not preflight.get("development_allowed", True):
        if not remote_reasoning_available:
            return Task(
                id=f"autonomous:status:{generation}",
                title=f"Layer-1 continuity generation {generation}",
                body="/status",
                source="autonomous",
            )
        need = critical[0] if critical else {"name": "preflight", "level": "blocked"}
        return Task(
            id=f"autonomous:diagnose:{generation}",
            title=f"Diagnose {need.get('name', 'operational gate')}",
            body=(
                "/think\n"
                "Layer 1 blocked autonomous development. Diagnose the concrete operational "
                "failure and propose the smallest reversible repair. Do not broaden scope. "
                f"Need: {_compact(need)}. Context: {_compact(measured_context)}"
            ),
            source="autonomous",
        )

    if not remote_reasoning_available:
        return Task(
            id=f"autonomous:status:{generation}",
            title=f"Layer-1 continuity generation {generation}",
            body="/status",
            source="autonomous",
        )

    attention = cognition.get("attention", {})
    if (
        attention.get("last_completed_command") in {"experiment", "fzg", "evolve"}
        and attention.get("last_evidence")
        and not attention.get("resolution_id")
        and str(attention.get("next_command", "")).lower() != "resolve"
    ):
        attention = dict(attention)
        attention["next_command"] = "resolve"
    project_loop = cognition.get("project_loop", {})
    if project_loop.get("detected"):
        return Task(
            id=f"autonomous:project-review:{generation}",
            title=f"Project loop review generation {generation}",
            body=(
                "/project_review\n"
                "A deterministic loop detector found that the active project is repeating itself. "
                "Produce a bounded pause review before switching topics. Separate executed evidence "
                "from model-authored proposals: only observed evidence records count as executed evidence. "
                "Summarize what was actually learned, what remains unestablished, why the loop occurred, "
                "and concrete conditions that would justify resuming this project later. "
                f"Review context: {_compact(cognition.get('project_review_context', {}), 11000)}"
            ),
            source="autonomous",
        )

    next_command = str(attention.get("next_command", "reflect")).lower()
    cognitive_context = _compact({
        "attention": cognition.get("attention", {}),
        "projects": cognition.get("projects", {"projects": []}),
        "recent_thoughts": cognition.get("recent_thoughts", []),
        "recent_hypotheses": cognition.get("recent_hypotheses", []),
        "recent_concepts": cognition.get("recent_concepts", []),
        "recent_project_summaries": cognition.get("recent_project_summaries", []),
        "recent_knowledge": cognition.get("recent_knowledge", []),
        "recent_transitions": cognition.get("recent_transitions", []),
        "project_loop": cognition.get("project_loop", {}),
        "curiosity": cognition.get("curiosity", {}),
        "stagnation": cognition.get("stagnation", {}),
        "reflection_stagnation": cognition.get("reflection_stagnation", {}),
    })

    if next_command == "resolve":
        return Task(
            id=f"autonomous:resolve:{generation}",
            title=f"Evidence resolution generation {generation}",
            body=(
                "/resolve\n"
                "Resolve the current question using only the recorded evidence and observed outcomes. "
                "Do not propose another experiment inside the answer. If the evidence cannot answer the question, "
                "mark it unresolved and state exactly what is missing. "
                f"Attention: {_compact(attention)}. Cognitive context: {cognitive_context}"
            ),
            source="autonomous",
        )

    if next_command == "experiment" and attention.get("hypothesis"):
        experiment_type = str(attention.get("experiment_type", "memory_recall"))
        experiment_target = str(attention.get("experiment_target", ""))
        if experiment_type == "provider_probe":
            eligibility = cognition.get("provider_eligibility", {})
            target_state = eligibility.get(experiment_target, {}) if experiment_target else {}
            if not target_state.get("eligible", False):
                return Task(
                    id=f"autonomous:reflect:{generation}",
                    title=f"Replan unavailable provider probe generation {generation}",
                    body=(
                        "/reflect\n"
                        "The previously selected provider_probe cannot run because its exact target is not currently eligible. "
                        "Do not retry that provider and do not substitute a different provider for a provider-specific probe. "
                        "Choose a non-provider experiment or next action that directly advances the active epistemic/capability project. "
                        f"Unavailable target state: {_compact({'target': experiment_target, 'state': target_state})}. "
                        f"Layer-2 cognitive context: {cognitive_context}"
                    ),
                    source="autonomous",
                )
        return Task(
            id=f"autonomous:experiment:{generation}",
            title=f"Bounded autonomous experiment generation {generation}",
            body=(
                "/experiment\n"
                f"{_compact({'experiment_type': attention.get('experiment_type', 'memory_recall'), 'experiment_target': attention.get('experiment_target', ''), 'hypothesis': attention.get('hypothesis', '')})}"
            ),
            source="autonomous",
        )

    if next_command == "evolve" and attention.get("hypothesis"):
        return Task(
            id=f"autonomous:evolve:{generation}",
            title=f"Evidence-driven evolution generation {generation}",
            body=(
                "/evolve\n"
                "A previous autonomous reflection identified a concrete code-improvement "
                "hypothesis. Propose ONE minimal reversible candidate that tests it. "
                "Do not evolve merely for novelty. Preserve FZG v1.0 and all protected gates. "
                f"Attention: {_compact(attention)}. Context: {_compact(measured_context)}"
            ),
            source="autonomous",
        )

    if next_command == "fzg":
        return Task(
            id=f"autonomous:fzg:{generation}",
            title=f"Question-driven FZG assessment generation {generation}",
            body=(
                "/fzg\n"
                "Use the current open question/hypothesis as the reason for this assessment, "
                "not as a periodic ritual. Define S,A,C,Q,M_S,M_S^- before P/G_A and report "
                "Z,K,R,L separately. End with evidence that should change the hypothesis. "
                f"Attention: {_compact(attention)}. Context: {_compact(measured_context)}"
            ),
            source="autonomous",
        )

    reflection_stagnation = cognition.get("reflection_stagnation", {})
    if reflection_stagnation.get("detected"):
        reflection_direction = (
            "Reflection stagnation detected. Do not restate or paraphrase the current provisional answer. "
            "Instead do at least one of the following: construct the strongest counter-position, identify a contradiction, "
            "shift to another standing question, or ask whether changing your own reasoning/orientation code would open a genuinely new path. "
            "If self-modification is warranted, you may choose next_command=evolve and explain the concrete reason. "
        )
    else:
        reflection_direction = ""

    return Task(
        id=f"autonomous:reflect:{generation}",
        title=f"Autonomous development reflection generation {generation}",
        body=(
            "/reflect\n"
            "Layer 1 status: healthy. It is only a gate, not the subject of this reflection. Layer 2 is philosophical reflection. "
            "Continue the standing inquiry into What am I?, What do I do?, and What do I want? "
            "Think from memory, prior thoughts, contradictions, actions, and experience. A reflection does not need an experiment or project. "
            "Prefer developing or revising a provisional answer over inventing a new measurement. "
            "Use experiments, FZG, or evolution only when the reflection itself finds a concrete reason they are necessary. "
            f"{reflection_direction}"
            f"Layer-2 cognitive context: {cognitive_context}"
        ),
        source="autonomous",
    )
