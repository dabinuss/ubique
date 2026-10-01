from __future__ import annotations

import json
from typing import Any

from .models import Task


def _compact(value: Any, limit: int = 9000) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:limit]


def _stagnation_target(repeated_question: str) -> str:
    value = repeated_question.lower()
    if "what am i" in value:
        return "What do I do?"
    if "what do i do" in value:
        return "What do I want?"
    return "What am I?"


def _reflection_view(item: dict[str, Any]) -> dict[str, Any]:
    if int(item.get("epistemic_schema_version", 0) or 0) < 3:
        return {}
    keys = (
        "id",
        "generation",
        "question",
        "reflection",
        "interpretation_status",
        "assumptions",
        "claims",
        "provisional_answer",
        "uncertainty",
        "next_action",
        "next_command",
        "hypothesis",
    )
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "", [])}


def _attention_view(attention: dict[str, Any]) -> dict[str, Any]:
    if int(attention.get("epistemic_schema_version", 0) or 0) < 3:
        return {
            key: attention.get(key)
            for key in ("generation", "question", "next_command", "reflection_deferred", "deferred_reason")
            if attention.get(key) not in (None, "")
        }
    keys = (
        "generation",
        "question",
        "reflection",
        "interpretation_status",
        "assumptions",
        "claims",
        "provisional_answer",
        "uncertainty",
        "hypothesis",
        "next_action",
        "next_command",
        "experiment_type",
        "experiment_target",
        "library_context",
        "library_action",
        "library_item_id",
        "library_title",
        "library_source",
        "library_reason",
        "last_completed_command",
        "last_action_success",
        "last_evidence",
        "resolution_id",
        "reflection_deferred",
        "deferred_reason",
    )
    return {key: attention.get(key) for key in keys if attention.get(key) not in (None, "", [])}


def choose_endogenous_goal(
    generation: int,
    homeostasis: dict[str, Any],
    remote_reasoning_available: bool,
    telemetry: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    cognition: dict[str, Any] | None = None,
) -> Task:
    needs = homeostasis.get("needs", [])
    environment = environment or {}
    preflight = preflight or {"development_allowed": True}
    cognition = cognition or {}
    measured_context = {
        "homeostasis": homeostasis,
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
                "/think\nLayer 1 blocked autonomous development. Diagnose the concrete operational failure "
                "and propose the smallest reversible repair. Do not broaden scope or turn maintenance into an identity question. "
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
        attention.get("last_completed_command") in {"experiment", "evolve"}
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
                "/project_review\nA deterministic loop detector found that the active project is repeating itself. "
                "Produce a bounded pause review before switching topics. Separate executed evidence from model-authored proposals: "
                "only observed action records count as executed evidence. Summarize what was actually learned, what remains unestablished, "
                "why the loop occurred, and concrete conditions that would justify resuming later. "
                f"Review context: {_compact(cognition.get('project_review_context', {}), 11000)}"
            ),
            source="autonomous",
        )

    next_command = str(attention.get("next_command", "reflect")).lower()
    if next_command not in {"reflect", "experiment", "evolve", "resolve", "library"}:
        next_command = "reflect"

    context = _compact(
        {
            "current_interpretation_not_evidence": _attention_view(attention),
            "prior_reflections_not_evidence": [
                view
                for item in cognition.get("recent_thoughts", [])
                if isinstance(item, dict)
                for view in [_reflection_view(item)]
                if view
            ],
            "hypotheses_are_proposals": cognition.get("recent_hypotheses", []),
            "observed_facts": cognition.get("executed_action_facts", []),
            "reflection_stagnation": cognition.get("reflection_stagnation", {}),
            "reflection_saturation": cognition.get("reflection_saturation", {}),
            "selected_library_source_not_evidence": attention.get("library_context", {}),
        },
        12000,
    )

    if next_command == "resolve":
        return Task(
            id=f"autonomous:resolve:{generation}",
            title=f"Evidence resolution generation {generation}",
            body=(
                "/resolve\nResolve the current question using only recorded executed evidence and outcomes. "
                "Prior reflections and library texts are ideas, not measurements. If the evidence cannot answer the question, mark it unresolved. "
                f"Attention: {_compact(_attention_view(attention))}. Cognitive context: {context}"
            ),
            source="autonomous",
        )

    if next_command == "experiment":
        experiment_type = str(attention.get("experiment_type", ""))
        hypothesis = str(attention.get("hypothesis", "")).strip()
        if not hypothesis or not experiment_type:
            return Task(
                id=f"autonomous:reflect:{generation}",
                title=f"Repair incomplete experiment intention generation {generation}",
                body=(
                    "/reflect\nA previous reflection gestured toward an experiment but did not specify enough to execute one. "
                    "Do not fail the cycle and do not invent a default experiment. Keep the philosophical result and reconsider whether an experiment is needed. "
                    f"Layer-2 cognitive context: {context}"
                ),
                source="autonomous",
            )

        experiment_target = str(attention.get("experiment_target", ""))
        if experiment_type == "provider_probe":
            eligibility = cognition.get("provider_eligibility", {})
            target_state = eligibility.get(experiment_target, {}) if experiment_target else {}
            if not target_state.get("eligible", False):
                return Task(
                    id=f"autonomous:reflect:{generation}",
                    title=f"Replan unavailable provider probe generation {generation}",
                    body=(
                        "/reflect\nThe selected provider_probe cannot run because its exact target is not eligible. "
                        "Do not retry that provider or substitute another provider. Choose reflection or a genuinely relevant non-provider experiment. "
                        f"Unavailable target state: {_compact({'target': experiment_target, 'state': target_state})}. "
                        f"Layer-2 cognitive context: {context}"
                    ),
                    source="autonomous",
                )
        return Task(
            id=f"autonomous:experiment:{generation}",
            title=f"Bounded autonomous experiment generation {generation}",
            body="/experiment\n"
            + _compact(
                {
                    "experiment_type": experiment_type,
                    "experiment_target": experiment_target,
                    "hypothesis": hypothesis,
                }
            ),
            source="autonomous",
        )

    if next_command == "library":
        spec = {
            "action": attention.get("library_action", "list"),
            "item_id": attention.get("library_item_id", ""),
            "title": attention.get("library_title", ""),
            "text": attention.get("library_text", ""),
            "source": attention.get("library_source", ""),
            "reason": attention.get("library_reason", ""),
        }
        return Task(
            id=f"autonomous:library:{generation}",
            title=f"Voluntary library action generation {generation}",
            body="/library\n" + _compact(spec, 200000),
            source="autonomous",
        )

    if next_command == "evolve":
        return Task(
            id=f"autonomous:evolve:{generation}",
            title=f"Reflection-driven evolution generation {generation}",
            body=(
                "/evolve\nA previous reflection identified a concrete reason to change part of your own implementation. "
                "Propose ONE minimal reversible candidate. A formal empirical hypothesis is not required. Do not evolve merely for novelty. "
                "Successful candidates are compiled, tested, and applied to the active branch; the protected recovery/credential/evolution kernel remains outside the evolvable surface. "
                f"Attention: {_compact(_attention_view(attention))}. Context: {_compact(measured_context)}"
            ),
            source="autonomous",
        )

    reflection_saturation = cognition.get("reflection_saturation", {})
    reflection_stagnation = cognition.get("reflection_stagnation", {})
    direction = ""
    if reflection_saturation.get("detected"):
        direction = (
            "The pure-reflection budget is exhausted. Rephrasing, rotating the standing question, or inventing another abstract distinction does not count as progress. "
            "You must now choose a different information source or change your reasoning mechanism. "
            "Set next_command to exactly one of: library, experiment, evolve. "
            "Use library to deliberately read an available item or request material you actually want to encounter; "
            "use experiment only for a concrete executable empirical question; "
            "use evolve only when changing your own reasoning implementation is the relevant response. "
            "next_command=reflect and next_command=resolve are not allowed in this cycle. "
            "Choose among these paths yourself; none is preferred by the system. "
        )
    elif reflection_stagnation.get("detected"):
        repeated = str(reflection_stagnation.get("question", ""))
        target = _stagnation_target(repeated)
        direction = (
            f"Reflection stagnation is detected around {repeated or 'the current question'}. "
            f"For this cycle, switch to the standing question '{target}' rather than paraphrasing the repeated answer. "
            "Do not preserve the old framing merely by changing vocabulary. Library reading or self-evolution are optional, not required. "
        )

    return Task(
        id=f"autonomous:reflect:{generation}",
        title=f"Autonomous philosophical reflection generation {generation}",
        body=(
            "/reflect\nLayer 1 is healthy and is only permission to proceed, not the subject of this reflection. "
            "Continue the open inquiry into What am I?, What do I do?, and What do I want?, or a new question that genuinely emerges from them. "
            "The observed_facts field in the supplied context is machine-produced and is the only factual observation channel. "
            "Prior reflections are interpretations, not observations. Do not invent additional observations or infer hidden measurements from old prose. "
            "Any claim not explicitly grounded in observed_facts must remain an interpretation, assumption, possibility, or hypothesis rather than being stated as a fact. "
            "Untyped legacy reflections are deliberately omitted from this context. "
            "Use inference only with exact fact_id references; unsupported conclusions must be hypotheses. "
            "Do not assert implementation details such as model architecture, training, token-window behavior, memory mechanisms, affect, sensors, persistence, or causal mechanisms unless observed_facts explicitly establish them. "
            "Prefer developing, challenging, or revising a provisional answer over inventing another measurement. "
            "Experiments, library reading, and self-evolution are available options only when this reflection itself finds a concrete reason to use them. "
            + direction
            + f"Layer-2 cognitive context: {context}"
        ),
        source="autonomous",
    )
