from __future__ import annotations

import json
from typing import Any

from .memory import append_memory_record, recent_memory_records
from .state import read_json


SAFE_EXPERIMENTS = {"provider_probe", "memory_recall", "memory_abstraction", "hypothesis_ablation", "state_consistency"}
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

    if kind == "hypothesis_ablation":
        attention = read_json("attention.json", {})
        question = str(attention.get("question", "")).strip() or str(hypothesis).strip()
        concepts = recent_memory_records("concepts.jsonl", 6)
        concept_payload = json.dumps(concepts[-6:], ensure_ascii=False)[:7000]

        base_instruction = (
            "Generate ONE falsifiable hypothesis for the supplied research question. "
            "Return strict JSON only: {\"hypothesis\":\"...\",\"rationale\":\"...\"}. "
            "Do not mention this experiment or claim evidence you do not have. "
            f"Question: {question[:3000]}"
        )
        treatment_instruction = (
            base_instruction
            + " Use the following persistent conceptual memory when it is genuinely relevant, "
            + "but do not force irrelevant concepts into the answer. Concepts: "
            + concept_payload
        )

        provider = target if target in SAFE_PROVIDERS else "gemini"
        baseline_result = router.generate_with(provider, base_instruction)
        treatment_result = router.generate_with(provider, treatment_instruction)

        def parse_hypothesis(text: str) -> dict[str, str]:
            raw = text.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("hypothesis_ablation result must be a JSON object")
            h = str(value.get("hypothesis", "")).strip()
            rationale = str(value.get("rationale", "")).strip()
            if not h:
                raise ValueError("hypothesis_ablation result missing hypothesis")
            return {"hypothesis": h[:3000], "rationale": rationale[:2000]}

        baseline = parse_hypothesis(baseline_result.text)
        treatment = parse_hypothesis(treatment_result.text)

        def tokens(value: str) -> set[str]:
            cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
            return {token for token in cleaned.split() if len(token) > 2}

        base_tokens = tokens(baseline["hypothesis"])
        treatment_tokens = tokens(treatment["hypothesis"])
        union = base_tokens | treatment_tokens
        overlap = base_tokens & treatment_tokens
        jaccard_similarity = (len(overlap) / len(union)) if union else 1.0
        lexical_divergence = 1.0 - jaccard_similarity

        concept_terms = set()
        for concept in concepts:
            concept_terms |= tokens(str(concept.get("name", "")))
            concept_terms |= tokens(str(concept.get("summary", "")))
        baseline_concept_overlap = len(base_tokens & concept_terms)
        treatment_concept_overlap = len(treatment_tokens & concept_terms)

        return {
            "kind": kind,
            "passed": True,
            "provider": baseline_result.provider,
            "question": question[:1200],
            "concept_count": len(concepts),
            "baseline": baseline,
            "with_concepts": treatment,
            "metrics": {
                "lexical_divergence": round(lexical_divergence, 4),
                "baseline_unique_terms": len(base_tokens),
                "with_concepts_unique_terms": len(treatment_tokens),
                "baseline_concept_overlap": baseline_concept_overlap,
                "with_concepts_concept_overlap": treatment_concept_overlap,
            },
            "identification": (
                "paired same-provider probe only; model sampling is not seed-controlled, so differences "
                "are evidence of sensitivity to conceptual memory, not a general causal estimate"
            ),
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
