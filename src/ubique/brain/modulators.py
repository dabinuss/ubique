from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class ModulatorState:
    novelty: float = 0.25
    surprise: float = 0.2
    uncertainty: float = 0.5
    salience: float = 0.2
    exploration: float = 0.45
    plasticity: float = 0.45
    energy: float = 0.85
    sleep_pressure: float = 0.1

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ModulatorState":
        value = value or {}
        defaults = cls()
        return cls(**{
            field: clamp(value.get(field, getattr(defaults, field)))
            for field in asdict(defaults)
        })

    def observe(
        self,
        *,
        novelty: float,
        surprise: float,
        uncertainty: float,
        salience: float,
        external_events: int = 0,
    ) -> None:
        alpha = 0.35
        self.novelty = clamp((1 - alpha) * self.novelty + alpha * novelty)
        self.surprise = clamp((1 - alpha) * self.surprise + alpha * surprise)
        self.uncertainty = clamp((1 - alpha) * self.uncertainty + alpha * uncertainty)
        self.salience = clamp((1 - alpha) * self.salience + alpha * salience)
        self.exploration = clamp(
            0.35 * self.novelty + 0.35 * self.uncertainty + 0.2 * self.surprise + 0.1
        )
        self.plasticity = clamp(
            0.25 + 0.35 * self.novelty + 0.25 * self.surprise + 0.15 * self.salience
        )
        wake_cost = 0.025 + 0.015 * min(max(external_events, 0), 4)
        self.energy = clamp(self.energy - wake_cost)
        self.sleep_pressure = clamp(self.sleep_pressure + 0.035 + 0.01 * external_events)

    def spend(self, amount: float) -> None:
        self.energy = clamp(self.energy - max(0.0, amount))
        self.sleep_pressure = clamp(self.sleep_pressure + max(0.0, amount) * 0.35)

    def rest(self, depth: float = 0.2) -> None:
        depth = clamp(depth)
        self.energy = clamp(self.energy + 0.25 + 0.35 * depth)
        self.salience = clamp(self.salience * (0.86 - 0.2 * depth))
        self.sleep_pressure = clamp(self.sleep_pressure * (0.7 - 0.35 * depth))

    def consolidate(self, depth: float = 0.5) -> None:
        depth = clamp(depth)
        self.energy = clamp(self.energy + 0.12 * depth)
        self.sleep_pressure = clamp(self.sleep_pressure * (0.45 - 0.2 * depth))
        self.plasticity = clamp(self.plasticity * 0.8 + 0.08)

    def as_dict(self) -> dict[str, float]:
        return {key: round(value, 4) for key, value in asdict(self).items()}
