from __future__ import annotations

from typing import Any

from .state import read_json, write_json, utc_now


def assess_preflight(
    generation: int,
    homeostasis: dict[str, Any],
    environment: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """Operational layer-1 gate: measure whether development may proceed."""
    needs = homeostasis.get("needs", [])
    critical = [n for n in needs if n.get("level") == "critical"]
    checks = [
        {
            "name": "runtime_state",
            "ok": isinstance(read_json("runtime.json", {}), dict),
            "evidence": "runtime state readable",
        },
        {
            "name": "homeostasis",
            "ok": not critical,
            "evidence": "no critical needs" if not critical else f"{len(critical)} critical need(s)",
        },
        {
            "name": "repository_branch",
            "ok": environment.get("branch") in {None, "", "main", "master"},
            "evidence": f"branch={environment.get('branch')}",
        },
        {
            "name": "recovery",
            "ok": int(recovery.get("action_count", 0)) >= 0,
            "evidence": f"recovery_actions={recovery.get('action_count', 0)}",
        },
    ]
    healthy = all(item["ok"] for item in checks)
    result = {
        "timestamp": utc_now(),
        "generation": generation,
        "healthy": healthy,
        "development_allowed": healthy,
        "checks": checks,
        "critical_needs": critical,
    }
    write_json("preflight.json", result)
    return result
