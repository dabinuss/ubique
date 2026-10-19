"""Provider-independent experiments over Ubique's own experience.

New experimental ideas are welcome. Unsupported labels become open probes
rather than being refused, and reported results distinguish observing from
confirming a theory.
"""
from __future__ import annotations
from .text_quality import content_terms
from .inquiry import compare_sources

def run_local_experiment(spec: dict, ctx: dict) -> dict:
    kind = str(spec.get("experiment_type") or spec.get("kind") or "association_probe")
    hypothesis = str(spec.get("hypothesis") or spec.get("question") or "").strip()
    generation = ctx.get("generation")
    episodes = [e for e in reversed(ctx["hippocampus"].all_episodes())
                if e.get("epistemic_status") == "source_passage"][:80]
    outcome = {"kind": kind, "hypothesis": hypothesis[:1200],
               "generation": generation, "method": "local_observation",
               "provider": "local", "confirmed": False,
               "status": "inconclusive", "evidence_ids": []}

    if kind in {"source_disagreement", "compare_sources", "counterexample", "contrast"}:
        for a in episodes[:20]:
            for b in episodes[:40]:
                result = compare_sources(a, b)
                if result:
                    outcome.update({
                        "status": "observed_comparison",
                        "evidence_ids": result["source_ids"],
                        "observation": result["passages"],
                        "question": result["question"],
                        "interpretation": "Potential contrast, not an established contradiction",
                    })
                    break
            if outcome["evidence_ids"]:
                break
    else:
        # The agent can supply *any* hypothesis/topic and perform a local
        # retrieval probe. Unknown experiment labels remain valid open probes.
        cues = content_terms(hypothesis or str(spec.get("target", "")), 8)
        if not cues and episodes:
            cues = content_terms(str(episodes[0].get("text", "")), 8)
        associations = ctx["cortex"].recall(cues, limit=12) if cues else []
        matching = [e for e in episodes if set(content_terms(
                    str(e.get("text", "")), 60)) & set(cues)]
        outcome.update({
            "status": "observed_associations" if associations or matching else "inconclusive",
            "cues": cues,
            "observation": [x.get("label") for x in associations[:8]],
            "evidence_ids": [e.get("id") for e in matching[:4] if e.get("id")],
            "interpretation": "Retrieved memories are leads, not proof of a hypothesis",
        })
    outcome["performed"] = True
    ctx["hippocampus"].journal.append("experiment_observation", outcome,
                                       generation=generation)
    return outcome
