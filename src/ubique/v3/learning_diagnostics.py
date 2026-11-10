"""Passive epistemic inventory for diagnosing learning quality.

These counters measure provenance and status, not correctness or consciousness.
They never veto exploratory ideas or alter the action policy.
"""
from __future__ import annotations


def summarize_claims(claims: dict) -> dict[str, int]:
    records = list(claims.values())
    return {
        "total": len(records),
        "with_evidence_refs": sum(bool(c.get("source_refs")) for c in records),
        "without_evidence_refs": sum(not c.get("source_refs") for c in records),
        "source_claims": sum(c.get("status") == "SOURCE_CLAIM" for c in records),
        "hypotheses": sum(c.get("status") == "HYPOTHESIS" for c in records),
        "corroborated_labels": sum(c.get("status") == "CORROBORATED" for c in records),
        "disputed_or_rejected": sum(c.get("status") in {"DISPUTED", "REJECTED"}
                                   for c in records),
    }
