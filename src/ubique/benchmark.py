from __future__ import annotations

import json

from .models import Task
from .planner import make_prompt, parse_task
from .providers.fallback import FallbackProvider
from .autonomy import autonomous_task


def run_benchmark() -> dict:
    """Lightweight advisory checks for core plumbing, not a philosophy score."""
    checks: list[tuple[str, bool]] = []

    p = parse_task(Task(id="1", title="x", body="/plan\nImplement A"))
    checks.append(("parse_plan", p.command == "plan" and p.payload == "Implement A"))

    p = parse_task(Task(id="2", title="x", body="/evolve\nImprove parser"))
    checks.append(("parse_evolve", p.command == "evolve" and "Improve parser" in p.payload))

    p = parse_task(Task(id="3", title="x", body="/library\n{\"action\":\"list\"}"))
    checks.append(("parse_library", p.command == "library" and "action" in p.payload))

    p = parse_task(Task(id="4", title="old", body="/fzg\nAnalyze"))
    checks.append(("legacy_fzg_not_first_class", p.command == "think"))

    prompt = make_prompt("reflect", "{}", [])
    checks.append(("no_normative_fzg_injection", "FZG v1.0 is the normative theoretical basis" not in prompt))
    checks.append(("optional_library_visible", "Optional library catalog" in prompt))

    fallback = FallbackProvider().generate("Task:\nhello")
    checks.append(("fallback_provider", "hello" in fallback.text))

    auto = autonomous_task(24, remote_reasoning_available=True)
    checks.append(("healthy_cycle_reflects", auto.body.startswith("/reflect")))

    auto = autonomous_task(25, remote_reasoning_available=False)
    checks.append(("no_remote_uses_status", auto.body.startswith("/status")))

    score = sum(1 for _, ok in checks if ok)
    return {
        "score": score,
        "max_score": len(checks),
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
    }


def main() -> None:
    result = run_benchmark()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["score"] == result["max_score"] else 1)


if __name__ == "__main__":
    main()
