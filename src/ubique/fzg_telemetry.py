from __future__ import annotations

from collections import Counter
from typing import Any

from .memory import recent_episodes
from .state import read_json, write_json, utc_now


def measure_fzg_telemetry() -> dict[str, Any]:
    """Empirical, context-bounded observables for FZG. Never a global score."""
    episodes = recent_episodes(100)
    providers = read_json("providers.json", {})
    runtime = read_json("runtime.json", {})

    successful = [e for e in episodes if e.get("success")]
    commands = {str(e.get("command")) for e in successful if e.get("command")}
    provider_paths = {str(e.get("provider")) for e in successful if e.get("provider")}
    task_sources = {
        str(e.get("task_id", "")).split(":", 1)[0]
        for e in successful if e.get("task_id")
    }

    # These are observable proxies, explicitly not scalar intelligence values.
    profile = {
        "Z": {
            "observable": len(commands),
            "basis": "distinct successful functional command states in retained episodes",
        },
        "K": {
            "observable": len(task_sources | provider_paths),
            "basis": "distinct observed task-source/provider context classes",
        },
        "R": {
            "observable": len({
                (str(e.get("command")), str(e.get("provider")))
                for e in successful
                if e.get("command") and e.get("provider")
            }),
            "basis": "distinct successful command-provider relations",
        },
        "L": {
            "observable": len({
                (str(e.get("task_id", "")).split(":", 1)[0], str(e.get("provider")), str(e.get("command")))
                for e in successful
                if e.get("provider") and e.get("command")
            }),
            "basis": "distinct observed successful source-provider-command solution paths",
        },
    }

    failures = Counter(str(e.get("command")) for e in episodes if not e.get("success"))
    result = {
        "timestamp": utc_now(),
        "generation": int(runtime.get("generation", 0)),
        "context": "current GitHub-hosted Ubique repository and retained episode window",
        "P": {
            "status": "proxy_only",
            "observable": bool(runtime.get("last_cycle_finished_at")),
            "basis": "continuity is evidenced by persisted generations, not treated as intelligence",
        },
        "G_A": {
            "status": "observational_proxy",
            "successful_tasks": len(successful),
            "failed_tasks": len(episodes) - len(successful),
            "ablation_required_for_causal_claim": True,
        },
        "I_C": profile,
        "failure_modes": dict(failures),
        "provider_count": len([p for p in providers if p != "fallback"]),
        "limitations": [
            "Z/K/R/L are observable diversity proxies, not normalized intelligence scores.",
            "G_A is observational unless the specific mechanism has a valid intervention.",
        ],
    }
    result["ablation"] = controlled_ablation_proxy()
    write_json("fzg_telemetry.json", result)
    return result


def controlled_ablation_proxy() -> dict[str, Any]:
    """Safe deterministic M_S vs M_S^- proxy over routing redundancy.

    It does not disable production providers. It compares the observed routing
    mechanism with a constructed single-path control from persisted evidence.
    """
    providers = read_json("providers.json", {})
    remote = [
        rec for name, rec in providers.items()
        if name != "fallback" and isinstance(rec, dict)
    ]
    healthy = sum(
        1 for rec in remote
        if not rec.get("disabled_until")
    )
    # A is deliberately a robustness objective, not ordinary availability.
    # Under the same hypothetical loss of one healthy provider, a redundant
    # router can retain a path only when >=2 healthy providers exist. A
    # single-provider control necessarily loses its sole path.
    observed_q = 1.0 if healthy >= 2 else 0.0
    ablated_q = 0.0
    return {
        "A": "retain remote reasoning availability after loss of one healthy provider",
        "C": "persisted provider ledger under a structural one-provider-loss perturbation",
        "healthy_remote_providers": healthy,
        "Q_M_S": observed_q,
        "Q_M_S_minus": ablated_q,
        "G_A_proxy": observed_q - ablated_q,
        "identification": (
            "structural robustness proxy only; no live provider was disabled and "
            "this is not a general causal estimate of task performance"
        ),
        "M_S": "configured multi-provider router",
        "M_S_minus": "single-provider structural control under the same one-provider-loss perturbation",
    }
