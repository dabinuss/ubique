from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from ..memory import MEMORY_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SelfModelStore:
    """History-derived self representation with explicit epistemic provenance."""

    def __init__(self, path: Path | None = None, limit: int = 500):
        self.path = path or (MEMORY_DIR / "self_model.jsonl")
        self.limit = max(50, int(limit))

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {"timestamp": utc_now(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if len(lines) > self.limit:
            self.path.write_text("\n".join(lines[-self.limit:]) + "\n", encoding="utf-8")
        return value

    def record_outcome(
        self,
        *,
        generation: int,
        action_kind: str,
        success: bool,
        evidence: str,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        statement = (
            f"At generation {generation}, this runtime executed '{action_kind}' "
            f"and the recorded outcome was {'successful' if success else 'unsuccessful'}."
        )
        return self._append({
            "generation": generation,
            "kind": "action_history",
            "statement": statement,
            "epistemic_status": "observed_history",
            "confidence": 1.0,
            "basis_episode_ids": [episode_id] if episode_id else [],
            "evidence": str(evidence)[:2000],
        })

    def record_interpretations(
        self,
        updates: Iterable[dict[str, Any]],
        *,
        generation: int,
        provider: str,
        basis_episode_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        out = []
        basis = [str(value) for value in basis_episode_ids if str(value).strip()][:12]
        for raw in updates:
            if not isinstance(raw, dict):
                continue
            statement = str(raw.get("statement", "")).strip()
            if not statement:
                continue
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            out.append(self._append({
                "generation": generation,
                "kind": "self_interpretation",
                "statement": statement[:3000],
                "epistemic_status": "model_interpretation",
                "confidence": round(confidence, 4),
                "provider": provider[:80],
                "basis_episode_ids": basis,
            }))
        return out

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-max(0, limit):]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
        return out
