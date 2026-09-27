from __future__ import annotations

from typing import Any

from .state import utc_now


CONTINUATION_COMMANDS = {"reflect", "experiment", "fzg", "evolve", "resolve"}


def decide_pulse(
    generation: int,
    preflight: dict[str, Any],
    homeostasis: dict[str, Any],
    attention: dict[str, Any],
    failed_tasks: int,
) -> dict[str, Any]:
    """Decide whether a completed beat should immediately schedule another beat."""
    next_command = str(attention.get("next_command", "")).strip().lower()
    development_allowed = bool(preflight.get("development_allowed", False))
    usable_remote = int(homeostasis.get("usable_remote_providers", 0) or 0)

    reasons: list[str] = []
    if failed_tasks:
        reasons.append("task_failure")
    if not development_allowed:
        reasons.append("layer1_blocked")
    if usable_remote <= 0:
        reasons.append("no_usable_remote_reasoning")
    if next_command not in CONTINUATION_COMMANDS:
        reasons.append("no_continuation_command")

    should_continue = not reasons
    return {
        "timestamp": utc_now(),
        "generation": generation,
        "mode": "active" if should_continue else "watchdog",
        "should_continue": should_continue,
        "next_command": next_command or None,
        "minimum_delay_seconds": 30,
        "reason": "continuation_ready" if should_continue else ",".join(reasons),
        "watchdog_schedule": "*/15 * * * *",
    }
