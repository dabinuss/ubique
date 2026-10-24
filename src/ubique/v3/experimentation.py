"""Source-grounded, revisable local experiments that work without API keys.

A retrieval is an observation, not confirmation. Experiments preserve competing
interpretations, evidence IDs, gaps and an actionable next question.
"""
from __future__ import annotations

from .text_quality import content_terms
from .inquiry import compare_sources
from .evidence import assess


def _source_matches(episodes: list[dict], hypothesis: str) -> list[dict]:
    cues = set(content_terms(hypothesis, 25))
    return sorted(episodes, key=lambda ep: (
        len(cues & set(content_terms(str(ep.get("text", ""))[:4000], 90))),
        len(str(ep.get("text", ""))),
    ), reverse=True)


def run_local_experiment(spec: dict, ctx: dict) -> dict:
    kind = str(spec.get("experiment_type") or spec.get("kind") or "association_probe")
    hypothesis = str(spec.get("hypothesis") or spec.get("question") or "").strip()
    generation = ctx.get("generation")
    episodes = [e for e in reversed(ctx["hippocampus"].all_episodes())
                if e.get("epistemic_status") == "source_passage"][:120]
    ranked = _source_matches(episodes, hypothesis)
    preferred = set(spec.get("source_ids") or [])
    if preferred:
        ranked.sort(key=lambda e: e.get("id") in preferred, reverse=True)

    outcome = {"kind": kind, "hypothesis": hypothesis[:1200],
               "generation": generation, "method": "source_grounded_local_probe",
               "provider": "local", "performed": True, "confirmed": False,
               "status": "inconclusive", "evidence_ids": [],
               "limitations": "Memory retrieval and textual overlap are not proof.",
               "alternative": "The apparent connection may be verbal rather than causal."}

    if kind in {"source_disagreement", "compare_sources", "counterexample", "contrast"}:
        for a in ranked[:24]:
            for b in ranked[:48]:
                if a.get("source") == b.get("source"):
                    continue
                result = compare_sources(a, b)
                if result is None:
                    continue
                assessments = [assess(hypothesis, [ep]).as_dict()
                               for ep in (a, b)] if hypothesis else []
                outcome.update({
                    "status": "observed_comparison",
                    "evidence_ids": result["source_ids"],
                    "observation": result["passages"],
                    "assessments": assessments,
                    "question": result["question"],
                    "next_question": "Which new independent source could distinguish these interpretations?",
                    "interpretation": "The passages raise a testable comparison; disagreement is not established.",
                })
                break
            if outcome["evidence_ids"]:
                break
    else:
        cues = content_terms(hypothesis or str(spec.get("target", "")), 8)
        if not cues and ranked:
            cues = content_terms(str(ranked[0].get("text", "")), 8)
        associations = ctx["cortex"].recall(cues, limit=12) if cues else []
        relevant = [e for e in ranked if set(content_terms(
                    str(e.get("text", ""))[:4000], 90)) & set(cues)][:4]
        reviews = [assess(hypothesis, [e]).as_dict() for e in relevant] if hypothesis else []
        outcome.update({
            "status": "observed_associations" if associations or relevant else "inconclusive",
            "cues": cues, "observation": [x.get("label") for x in associations[:8]],
            "evidence_ids": [e["id"] for e in relevant if e.get("id")],
            "assessments": reviews,
            "next_question": "What observation could disconfirm the proposed link?",
            "interpretation": "Retrieved memories are leads, not proof of a hypothesis.",
        })
    ctx["hippocampus"].journal.append("experiment_observation", outcome,
                                       generation=generation)
    return outcome
