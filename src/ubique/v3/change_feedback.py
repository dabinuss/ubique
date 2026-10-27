"""Passive post-change observations: code merged is not automatically useful.

Effect review does not veto evolution or call a benchmark. It records whether
the modified cognitive path was exercised and whether its behavior changed.
An observed correlation is never promoted to proof of causation.
"""
from __future__ import annotations

MARKERS = {
    "semantics.py": "association_hypothesis",
    "inquiry.py": "inquiry",
    "curiosity.py": "interest",
    "book_discovery.py": "library_discovery",
    "workspace.py": "action_outcome",
    "planner.py": "action_outcome",
    "actions.py": "action_outcome",
    "experimentation.py": "experiment_observation",
}


def review_changes(journal, generation: int, *, observation_window: int = 24) -> dict | None:
    """Review a past successful self-edit after enough actual runtime pulses."""
    if generation < observation_window * 2:
        return None
    accepted = [
        ev for ev in journal.recent("change_outcome", limit=12)
        if ev.get("data", {}).get("success") is True
        and ev.get("generation") is not None
        and int(ev["generation"]) + observation_window <= generation
    ]
    if not accepted:
        return None
    history = journal.events()
    reviewed = {
        str(ev.get("data", {}).get("change_id"))
        for ev in history if ev.get("kind") == "change_review"
    }
    for change in reversed(accepted):
        identifier = str(change.get("id", ""))
        if not identifier or identifier in reviewed:
            continue
        generation_changed = int(change["generation"])
        target = str(change.get("data", {}).get("intent", {}).get("target", ""))
        marker = next((value for suffix, value in MARKERS.items()
                       if target.endswith(suffix)), "action_outcome")
        prior = [
            ev for ev in history
            if ev.get("kind") == marker
            and ev.get("generation") is not None
            and generation_changed - observation_window <= int(ev["generation"]) < generation_changed
        ]
        after = [
            ev for ev in history
            if ev.get("kind") == marker
            and ev.get("generation") is not None
            and generation_changed < int(ev["generation"]) <= generation_changed + observation_window
        ]
        # A code change that was never exercised cannot be pronounced effective.
        if not after:
            verdict = "not_observed"
        elif len(after) == len(prior):
            verdict = "activity_unchanged"
        else:
            verdict = "activity_changed"
        record = {
            "change_id": identifier,
            "target": target,
            "generation_changed": generation_changed,
            "window_generations": observation_window,
            "observed_event_kind": marker,
            "events_before": len(prior),
            "events_after": len(after),
            "verdict": verdict,
            "interpretation": (
                "Observed event frequency only. Neither successful behavior "
                "nor causation is established by this signal."
            ),
        }
        journal.append("change_review", record, generation=generation)
        return record
    return None
