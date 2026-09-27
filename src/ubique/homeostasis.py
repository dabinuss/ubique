from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from .memory import episode_count, recent_episodes
from .state import read_json, write_json, utc_now


@dataclass(slots=True)
class Need:
    name: str
    level: str
    value: float
    target: str
    evidence: str


def _recent_success_ratio(episodes: list[dict[str, Any]]) -> float:
    if not episodes:
        return 1.0
    return sum(1 for e in episodes if e.get("success")) / len(episodes)


def assess_homeostasis(
    memory_limit: int = 500,
    configured_remote: list[str] | None = None,
) -> dict[str, Any]:
    """Assess operational needs without collapsing them into one score."""
    episodes = recent_episodes(24)
    providers = read_json("providers.json", {})
    runtime = read_json("runtime.json", {})

    success_ratio = _recent_success_ratio(episodes)
    retained_episodes = episode_count()
    configured_remote = configured_remote or [
        name for name in providers if name != "fallback"
    ]
    usable_remote = 0
    now = datetime.now(timezone.utc)
    for name in configured_remote:
        rec = providers.get(name, {})
        until = rec.get("disabled_until") if isinstance(rec, dict) else None
        if not until:
            usable_remote += 1
            continue
        try:
            if datetime.fromisoformat(until) <= now:
                usable_remote += 1
        except (TypeError, ValueError):
            usable_remote += 1

    needs = [
        Need(
            "cycle_reliability",
            "critical" if success_ratio < 0.5 else "watch" if success_ratio < 0.8 else "stable",
            success_ratio,
            ">=0.80 recent successful tasks",
            f"{sum(1 for e in episodes if e.get('success'))}/{len(episodes) or 0} recent tasks succeeded",
        ),
        Need(
            "reasoning_redundancy",
            "critical" if usable_remote == 0 else "watch" if usable_remote == 1 else "stable",
            float(usable_remote),
            ">=2 independently configured remote reasoning paths",
            f"{usable_remote} remote providers currently usable according to the persisted ledger",
        ),
        Need(
            "memory_pressure",
            "critical" if retained_episodes >= memory_limit else "watch" if retained_episodes >= int(memory_limit * 0.8) else "stable",
            float(retained_episodes),
            f"<{memory_limit} retained episodes",
            f"{retained_episodes}/{memory_limit} retained episode records",
        ),
    ]

    snapshot = {
        "timestamp": utc_now(),
        "generation": int(runtime.get("generation", 0)),
        "needs": [asdict(n) for n in needs],
        "recent_success_ratio": success_ratio,
        "retained_episode_count": retained_episodes,
        "configured_remote_providers": sorted(configured_remote),
        "usable_remote_providers": usable_remote,
    }
    write_json("homeostasis.json", snapshot)
    return snapshot


def highest_need(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    rank = {"critical": 2, "watch": 1, "stable": 0}
    needs = list(snapshot.get("needs", []))
    if not needs:
        return None
    needs.sort(key=lambda x: (rank.get(str(x.get("level")), 0), -float(x.get("value", 0))), reverse=True)
    return needs[0]
