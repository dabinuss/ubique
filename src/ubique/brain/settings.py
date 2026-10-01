from __future__ import annotations

from dataclasses import dataclass
import os


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(value, high))


@dataclass(frozen=True, slots=True)
class BrainSettings:
    workspace_slots: int = 7
    max_substrates: int = 2
    max_immediate_pulses: int = 3
    activation_threshold: float = 0.24

    @classmethod
    def from_env(cls) -> "BrainSettings":
        try:
            threshold = float(os.getenv("UBIQUE_BRAIN_ACTIVATION_THRESHOLD", "0.24"))
        except ValueError:
            threshold = 0.24
        return cls(
            workspace_slots=_int("UBIQUE_BRAIN_WORKSPACE_SLOTS", 7, 3, 12),
            max_substrates=_int("UBIQUE_BRAIN_MAX_SUBSTRATES", 2, 1, 3),
            max_immediate_pulses=_int("UBIQUE_BRAIN_MAX_IMMEDIATE_PULSES", 3, 1, 8),
            activation_threshold=max(0.05, min(threshold, 0.9)),
        )
