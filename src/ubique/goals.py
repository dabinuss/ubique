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
    next_command = str(attention.get("next_command", "reflect")).lower()
    cognitive_context = _compact(cognition)

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

    return Task(
        id=f"autonomous:reflect:{generation}",
        title=f"Autonomous development reflection generation {generation}",
        body=(
            "/reflect\n"
            "Layer 1 is healthy, so Layer 2 must make epistemic or capability progress. "
            "Continue an active project when useful; otherwise identify a new concrete unknown, "
            "weakness, opportunity or contradiction. Avoid status-only output. Use prior thoughts "
            "and hypotheses so the process continues across generations. "
            f"Measured context: {_compact(measured_context)}. Cognitive memory: {cognitive_context}"
        ),
        source="autonomous",
    )
