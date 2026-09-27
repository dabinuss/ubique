from __future__ import annotations

import json
from typing import Any

from .memory import recent_memory_records
from .state import read_json


SAFE_EXPERIMENTS = {"provider_probe", "memory_recall", "state_consistency"}
SAFE_PROVIDERS = {"gemini", "groq", "huggingface"}


def run_experiment(kind: str, target: str, router: Any, hypothesis: str) -> dict[str, Any]:
    if kind not in SAFE_EXPERIMENTS:
        raise ValueError(f"unsupported experiment: {kind}")

    if kind == "provider_probe":
        provider = target if target in SAFE_PROVIDERS else "groq"
        prompt = (
            "This is a bounded provider-path validation for Ubique. "
            "Return exactly the token UBIQUE_PROBE_OK and nothing else. "
            f"Hypothesis under test: {hypothesis[:800]}"
        )
        result = router.generate_with(provider, prompt)
        passed = result.text.strip() == "UBIQUE_PROBE_OK"
        return {"kind": kind, "target": provider, "passed": passed, "provider": result.provider, "evidence": result.text[:200]}

    if kind == "memory_recall":
        thoughts = recent_memory_records("thoughts.jsonl", 5)
        hypotheses = recent_memory_records("hypotheses.jsonl", 5)
        passed = bool(thoughts and hypotheses)
        return {"kind": kind, "passed": passed, "thought_count": len(thoughts), "hypothesis_count": len(hypotheses)}

    attention = read_json("attention.json", {})
    projects = read_json("projects.json", {"projects": []})
    passed = isinstance(attention, dict) and isinstance(projects.get("projects", []), list)
    return {"kind": kind, "passed": passed, "attention_keys": sorted(attention.keys()), "project_count": len(projects.get("projects", []))}
