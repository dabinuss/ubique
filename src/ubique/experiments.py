from __future__ import annotations

import json
from typing import Any

from .memory import append_memory_record, recent_memory_records
from .state import read_json


SAFE_EXPERIMENTS = {"provider_probe", "memory_recall", "memory_abstraction", "state_consistency"}
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

    if kind == "memory_abstraction":
        thoughts = recent_memory_records("thoughts.jsonl", 8)
        hypotheses = recent_memory_records("hypotheses.jsonl", 8)
        source = {"thoughts": thoughts[-6:], "hypotheses": hypotheses[-6:]}
        prompt = (
            "Synthesize reusable conceptual memory from the supplied Ubique thought/hypothesis records. "
            "Return strict JSON only with schema "
            "{\"concepts\":[{\"name\":\"...\",\"summary\":\"...\",\"supports\":[\"thought:id\"],"
            "\"exceptions\":[\"...\"],\"open_question\":\"...\"}]}. "
            "Create at most 4 concepts. Do not invent evidence not present in the records. "
            f"Records: {json.dumps(source, ensure_ascii=False)[:9000]}"
        )
        result = router.generate(prompt)
        raw = result.text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        data = json.loads(raw)
        concepts = data.get("concepts", []) if isinstance(data, dict) else []
        accepted = 0
        for concept in concepts[:4]:
            if not isinstance(concept, dict):
                continue
            name = str(concept.get("name", "")).strip()
            summary = str(concept.get("summary", "")).strip()
            if not name or not summary:
                continue
            append_memory_record(
                "concepts.jsonl",
                {
                    "name": name[:200],
                    "summary": summary[:2000],
                    "supports": concept.get("supports", [])[:12] if isinstance(concept.get("supports", []), list) else [],
                    "exceptions": concept.get("exceptions", [])[:8] if isinstance(concept.get("exceptions", []), list) else [],
                    "open_question": str(concept.get("open_question", ""))[:1200],
                    "source_provider": result.provider,
                },
                limit=300,
            )
            accepted += 1
        return {
            "kind": kind,
            "passed": accepted > 0,
            "provider": result.provider,
            "concepts_created": accepted,
            "source_thoughts": len(thoughts),
            "source_hypotheses": len(hypotheses),
        }

    if kind == "memory_recall":
        thoughts = recent_memory_records("thoughts.jsonl", 5)
        hypotheses = recent_memory_records("hypotheses.jsonl", 5)
        passed = bool(thoughts and hypotheses)
        return {"kind": kind, "passed": passed, "thought_count": len(thoughts), "hypothesis_count": len(hypotheses)}

    attention = read_json("attention.json", {})
    projects = read_json("projects.json", {"projects": []})
    passed = isinstance(attention, dict) and isinstance(projects.get("projects", []), list)
    return {"kind": kind, "passed": passed, "attention_keys": sorted(attention.keys()), "project_count": len(projects.get("projects", []))}
