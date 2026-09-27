from __future__ import annotations

import json

from .models import Task
from .planner import parse_task
from .providers.fallback import FallbackProvider
from .autonomy import autonomous_task


def run_benchmark() -> dict:
    checks: list[tuple[str, bool]] = []

    p = parse_task(Task(id="1", title="x", body="/plan\nImplement A"))
    checks.append(("parse_plan", p.command == "plan" and p.payload == "Implement A"))

    p = parse_task(Task(id="2", title="x", body="/evolve\nImprove parser"))
    checks.append(("parse_evolve", p.command == "evolve" and "Improve parser" in p.payload))

    p = parse_task(Task(id="3", title="Fallback title", body=""))
    checks.append(("empty_body", p.command == "think" and p.payload == "Fallback title"))

    fallback = FallbackProvider().generate("Task:\nhello")
    checks.append(("fallback_provider", "hello" in fallback.text))

    auto = autonomous_task(3, remote_reasoning_available=True)
    checks.append(("autonomous_evolution_cycle", auto.body.startswith("/evolve")))

    auto = autonomous_task(4, remote_reasoning_available=True)
    checks.append(("autonomous_observation_cycle", auto.body.startswith("/fzg")))

    score = sum(1 for _, ok in checks if ok)
    return {"score": score, "max_score": len(checks),
            "checks": [{"name": name, "ok": ok} for name, ok in checks]}


def main() -> None:
    result = run_benchmark()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["score"] == result["max_score"] else 1)


if __name__ == "__main__":
    main()
