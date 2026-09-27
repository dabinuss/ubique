from __future__ import annotations

from typing import Any

from .models import Task


def choose_endogenous_goal(
    generation: int,
    homeostasis: dict[str, Any],
    remote_reasoning_available: bool,
) -> Task:
    """Choose one operational goal from measured state, not a user prompt."""
    needs = homeostasis.get("needs", [])
    critical = [n for n in needs if n.get("level") == "critical"]
    watch = [n for n in needs if n.get("level") == "watch"]

    if remote_reasoning_available and critical:
        need = critical[0]
        return Task(
            id=f"autonomous:diagnose:{generation}",
            title=f"Diagnose {need.get('name', 'critical need')}",
            body=(
                "/think\n"
                "This goal was selected endogenously from Ubique homeostasis. "
                f"Need: {need}. Diagnose the concrete failure mode from recent memory and "
                "propose the smallest reversible action that improves the measured target. "
                "Do not change FZG definitions and do not claim improvement without evidence."
            ),
            source="autonomous",
        )

    # Expensive self-modification only when no critical need is unresolved.
    if remote_reasoning_available and generation % 24 == 0:
        return Task(
            id=f"autonomous:evolve:{generation}",
            title=f"Endogenous evolution generation {generation}",
            body=(
                "/evolve\n"
                "Select ONE small, testable improvement from recent memory, homeostatic needs, "
                "FZG telemetry and provider observations. Improve reliable capability, adaptation, "
                "observability, resource efficiency or solution-path diversity. Preserve FZG v1.0 "
                "and the recovery kernel. Return only strict evolution JSON."
            ),
            source="autonomous",
        )

    if remote_reasoning_available and (watch or generation % 6 == 0):
        return Task(
            id=f"autonomous:fzg:{generation}",
            title=f"Endogenous FZG assessment generation {generation}",
            body=(
                "/fzg\n"
                "Perform an endogenous FZG v1.0 analysis using current homeostasis, recent "
                "episodes, provider state and empirical telemetry. Define S,A,C,Q,M_S,M_S^- "
                "before P/G_A; report Z,K,R,L separately and identify the next measurable gap."
            ),
            source="autonomous",
        )

    return Task(
        id=f"autonomous:status:{generation}",
        title=f"Autonomous continuity generation {generation}",
        body="/status",
        source="autonomous",
    )
