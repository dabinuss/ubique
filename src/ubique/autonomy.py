from __future__ import annotations

from typing import Any

from .goals import choose_endogenous_goal
from .models import Task


def autonomous_task(
    generation: int,
    remote_reasoning_available: bool,
    homeostasis: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> Task:
    """Choose Ubique's next endogenous action from state plus bounded cadence."""
    snapshot = homeostasis or {"needs": []}
    return choose_endogenous_goal(
        generation=generation,
        homeostasis=snapshot,
        remote_reasoning_available=remote_reasoning_available,
        telemetry=telemetry,
        environment=environment,
    )
