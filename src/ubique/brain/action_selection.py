from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .modulators import ModulatorState, clamp


ALLOWED_ACTIONS = {
    "rest",
    "attend",
    "library",
    "experiment",
    "evolve",
    "consolidate",
}

SAFE_EXPERIMENTS = {
    "provider_probe",
    "memory_recall",
    "memory_abstraction",
    "hypothesis_ablation",
    "state_consistency",
}
SAFE_LIBRARY_ACTIONS = {"list", "read", "add", "request", "note"}


def _payload_is_executable(kind: str, payload: dict[str, Any]) -> bool:
    """Reject incomplete model proposals before they enter action competition."""
    if kind == "experiment":
        return str(payload.get("experiment_type", "")).strip() in SAFE_EXPERIMENTS
    if kind == "library":
        action = str(payload.get("action", "")).strip().lower()
        if action not in SAFE_LIBRARY_ACTIONS:
            return False
        if action in {"read", "note"} and not str(payload.get("item_id", "")).strip():
            return False
        if action in {"add", "request"} and not str(payload.get("title", "")).strip():
            return False
        if action == "note" and not str(payload.get("text", "")).strip():
            return False
    return True


@dataclass(slots=True)
class ActionCandidate:
    kind: str
    description: str
    support: float = 0.5
    utility: float = 0.5
    information_gain: float = 0.4
    novelty: float = 0.3
    energy_cost: float = 0.3
    source: str = "endogenous"
    payload: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class ActionSelector:
    """Basal-ganglia-inspired competition between possible actions."""

    def baseline_candidates(
        self,
        modulators: ModulatorState,
        *,
        memory_count: int,
        workspace_activation: float,
    ) -> list[ActionCandidate]:
        quiet = clamp(1.0 - workspace_activation)
        rest_support = clamp(
            0.34 * quiet
            + 0.36 * (1.0 - modulators.energy)
            + 0.30 * modulators.sleep_pressure
        )
        consolidate_support = clamp(
            0.58 * modulators.sleep_pressure
            + 0.2 * (1.0 - modulators.energy)
            + 0.22 * min(1.0, memory_count / 30.0)
        )
        attend_support = clamp(
            0.4 * modulators.salience
            + 0.25 * modulators.uncertainty
            + 0.2 * modulators.novelty
            + 0.15 * workspace_activation
        )
        return [
            ActionCandidate(
                kind="rest",
                description="Allow activation to decay without forcing cognition.",
                support=rest_support,
                utility=0.45 + 0.35 * (1.0 - modulators.energy),
                information_gain=0.0,
                novelty=0.0,
                energy_cost=0.0,
                source="homeostasis",
            ),
            ActionCandidate(
                kind="consolidate",
                description="Replay and consolidate existing experience offline.",
                support=consolidate_support,
                utility=0.55,
                information_gain=0.35,
                novelty=0.15,
                energy_cost=0.12,
                source="homeostasis",
                payload={
                    "mode": "rem"
                    if modulators.exploration > 0.72 and modulators.energy > 0.3
                    else "nrem"
                },
            ),
            ActionCandidate(
                kind="attend",
                description="Continue processing the active workspace without external action.",
                support=attend_support,
                utility=0.5,
                information_gain=0.5 * modulators.uncertainty,
                novelty=modulators.novelty,
                energy_cost=0.16,
                source="workspace",
            ),
        ]

    def from_model_actions(
        self,
        actions: Iterable[dict[str, Any]],
        *,
        source: str,
    ) -> list[ActionCandidate]:
        out: list[ActionCandidate] = []
        for raw in actions:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind", "")).strip().lower()
            if kind not in ALLOWED_ACTIONS:
                continue
            description = str(raw.get("description", "")).strip()
            if not description:
                continue
            payload = raw.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            if not _payload_is_executable(kind, payload):
                continue
            out.append(ActionCandidate(
                kind=kind,
                description=description[:1600],
                support=clamp(raw.get("support", raw.get("utility", 0.5))),
                utility=clamp(raw.get("utility", 0.5)),
                information_gain=clamp(raw.get("information_gain", 0.4)),
                novelty=clamp(raw.get("novelty", 0.3)),
                energy_cost=clamp(raw.get("energy_cost", 0.25)),
                source=source[:160],
                payload=payload,
            ))
        return out

    @staticmethod
    def _score(candidate: ActionCandidate, modulators: ModulatorState) -> float:
        score = (
            0.27 * clamp(candidate.support)
            + 0.21 * clamp(candidate.utility)
            + 0.17 * clamp(candidate.information_gain) * (0.6 + 0.4 * modulators.uncertainty)
            + 0.11 * clamp(candidate.novelty) * (0.55 + 0.45 * modulators.exploration)
            + 0.12 * modulators.salience
            + 0.07 * modulators.exploration
            - 0.12 * clamp(candidate.energy_cost) * (1.0 - 0.45 * modulators.energy)
        )
        if candidate.kind == "rest":
            score += 0.18 * (1.0 - modulators.energy) + 0.1 * modulators.sleep_pressure
        elif candidate.kind == "consolidate":
            score += 0.19 * modulators.sleep_pressure
        elif candidate.kind == "evolve":
            score -= 0.08

        if candidate.kind not in {"rest", "consolidate"} and modulators.energy < 0.28:
            score -= (0.28 - modulators.energy) * 1.8
        return score

    @staticmethod
    def _habituation_penalty(kind: str, recent_actions: Iterable[str]) -> float:
        """Reduce the pull of an action that has just been selected repeatedly.

        This is deliberately content-agnostic: the runtime should become less
        responsive to any repeated action pattern, not to a hard-coded topic.
        Rest is only weakly habituated because sustained low-energy states may
        legitimately require repeated rest.
        """
        history = [str(value).strip().lower() for value in recent_actions if str(value).strip()][-8:]
        if not history:
            return 0.0

        count = sum(1 for value in history if value == kind)
        streak = 0
        for value in reversed(history):
            if value != kind:
                break
            streak += 1

        if count == 0:
            return 0.0

        penalty = min(0.32, 0.03 * count + 0.04 * streak)
        if kind == "rest":
            penalty *= 0.3
        elif kind in {"consolidate", "attend"}:
            penalty *= 1.15
        return min(0.36, penalty)

    def select(
        self,
        candidates: Iterable[ActionCandidate],
        modulators: ModulatorState,
        recent_actions: Iterable[str] = (),
    ) -> tuple[ActionCandidate, list[ActionCandidate]]:
        ranked: list[ActionCandidate] = []
        metabolic_lock = modulators.energy < 0.15
        history = list(recent_actions)
        for candidate in candidates:
            if candidate.kind not in ALLOWED_ACTIONS:
                continue
            if metabolic_lock and candidate.kind not in {"rest", "consolidate"}:
                continue
            raw_score = self._score(candidate, modulators)
            candidate.score = round(
                raw_score - self._habituation_penalty(candidate.kind, history),
                5,
            )
            ranked.append(candidate)
        if not ranked:
            ranked = [ActionCandidate(kind="rest", description="No action candidates.")]
            ranked[0].score = self._score(ranked[0], modulators)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[0], ranked
