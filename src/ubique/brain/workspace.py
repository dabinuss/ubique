from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class WorkspaceItem:
    id: str
    kind: str
    label: str
    activation: float
    salience: float = 0.0
    novelty: float = 0.0
    surprise: float = 0.0
    epistemic_status: str = "unknown"
    source: str = ""

    @property
    def ignition_score(self) -> float:
        return (
            0.5 * _clamp(self.activation)
            + 0.25 * _clamp(self.salience)
            + 0.13 * _clamp(self.novelty)
            + 0.12 * _clamp(self.surprise)
        )


class GlobalWorkspace:
    """Small competitive workspace rebuilt on every pulse."""

    def __init__(self, capacity: int = 7):
        self.capacity = max(3, min(int(capacity), 12))
        self.items: list[WorkspaceItem] = []

    def compete(self, candidates: Iterable[dict[str, Any]]) -> list[WorkspaceItem]:
        best: dict[str, WorkspaceItem] = {}
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label", "")).strip()
            item_id = str(raw.get("id", "")).strip() or label.lower()
            if not label or not item_id:
                continue
            item = WorkspaceItem(
                id=item_id[:200],
                kind=str(raw.get("kind", "concept"))[:60],
                label=label[:500],
                activation=_clamp(raw.get("activation", 0.0)),
                salience=_clamp(raw.get("salience", 0.0)),
                novelty=_clamp(raw.get("novelty", 0.0)),
                surprise=_clamp(raw.get("surprise", 0.0)),
                epistemic_status=str(raw.get("epistemic_status", "unknown"))[:80],
                source=str(raw.get("source", ""))[:160],
            )
            previous = best.get(item.id)
            if previous is None or item.ignition_score > previous.ignition_score:
                best[item.id] = item

        ranked = sorted(
            best.values(),
            key=lambda item: item.ignition_score,
            reverse=True,
        )
        self.items = ranked[: self.capacity]
        return list(self.items)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                **asdict(item),
                "ignition_score": round(item.ignition_score, 4),
            }
            for item in self.items
        ]

    def mean_activation(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.activation for item in self.items) / len(self.items)

    def prompt_view(self) -> str:
        if not self.items:
            return "- workspace quiet"
        lines = []
        for item in self.items:
            lines.append(
                f"- [{item.kind}/{item.epistemic_status}] {item.label} "
                f"(activation={item.activation:.2f}, salience={item.salience:.2f}, "
                f"novelty={item.novelty:.2f})"
            )
        return "\n".join(lines)
