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
    preflight: dict[str, Any] | None = None,
    cognition: dict[str, Any] | None = None,
) -> Task:
    """Choose the next endogenous action from layer-1 health and layer-2 cognition."""
    snapshot = homeostasis or {"needs": []}
    return choose_endogenous_goal(
        generation=generation,
        homeostasis=snapshot,
        remote_reasoning_available=remote_reasoning_available,
        telemetry=telemetry,
        environment=environment,
        preflight=preflight,
        cognition=cognition,
    )
