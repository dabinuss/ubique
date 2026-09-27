from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .state import read_json, write_json, utc_now


def perform_recovery() -> dict[str, Any]:
    """Apply only deterministic, reversible repairs to persisted operational state."""
    providers = read_json("providers.json", {})
    now = datetime.now(timezone.utc)
    cleared: list[str] = []

    for name, rec in providers.items():
        if not isinstance(rec, dict):
            continue
        until = rec.get("disabled_until")
        if not until:
            continue
        try:
            if datetime.fromisoformat(until) <= now:
                rec["disabled_until"] = None
                cleared.append(name)
        except (TypeError, ValueError):
            rec["disabled_until"] = None
            cleared.append(name)

    write_json("providers.json", providers)
    result = {
        "timestamp": utc_now(),
        "cleared_expired_provider_backoffs": cleared,
        "action_count": len(cleared),
    }
    write_json("recovery.json", result)
    return result
