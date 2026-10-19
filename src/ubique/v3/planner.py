"""Procedural action memory: learn from observed outcomes rather than rigid scheduling."""
from __future__ import annotations
from collections import defaultdict
from .actions import Candidate
from .storage import Journal

class Planner:
    def __init__(self, journal: Journal):
        self.journal = journal
        self.quality = defaultdict(lambda: [0, 0])
        for event in journal.events():
            if event["kind"] == "action_outcome":
                d = event.get("data", {})
                if d.get("kind"):
                    self.quality[d["kind"]][0] += int(bool(d.get("success")))
                    self.quality[d["kind"]][1] += 1

    def rank(self, candidates: list[Candidate], *, last: list[str], urgency: float = 0.5):
        seen = set()
        for item in candidates:
            wins, total = self.quality[item.kind]
            # Reliability bonus (Laplace smoothing)
            if total:
                reliability = (wins + 1) / (total + 2)
                item.score += 0.14 * (reliability - 0.5)
            else:
                # Curiosity boost: never‑seen kinds get a small positive nudge
                item.score += 0.10
            # Penalty for recent repetition
            item.score -= 0.025 * last[-10:].count(item.kind)
            # Penalty for identical payload seen in this ranking pass
            payload_marker = (item.kind, str(item.payload))
            if payload_marker in seen:
                item.score -= 0.15
            seen.add(payload_marker)
        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def record(self, kind: str, success: bool) -> None:
        self.quality[kind][0] += int(success)
        self.quality[kind][1] += 1
