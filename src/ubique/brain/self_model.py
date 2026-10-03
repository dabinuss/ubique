from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

from ..memory import MEMORY_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def _position_similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _topic_key(topic: str) -> str:
    words = sorted(_tokens(topic))
    return "-".join(words[:12])[:180] or "unspecified"


class SelfModelStore:
    """History-derived self representation with explicit epistemic provenance.

    The store distinguishes transient model interpretations from established,
    revisable self-positions. A single model output never becomes personality:
    a stance must recur across generations and be grounded in at least one
    observed/external source before it is exposed as an established stance.
    """

    def __init__(self, path: Path | None = None, limit: int = 500):
        self.path = path or (MEMORY_DIR / "self_model.jsonl")
        self.limit = max(50, int(limit))

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
        return out

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {"timestamp": utc_now(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

        records = self._read_records()
        if len(records) > self.limit:
            # Routine action history must not erase established personality.
            # Preserve the latest mature stance for every topic, then spend the
            # remaining capacity on recent history/candidates.
            latest_stances: dict[str, dict[str, Any]] = {}
            for item in records:
                if item.get("kind") != "stance":
                    continue
                key = str(item.get("topic_key", "")).strip()
                if key:
                    latest_stances[key] = item

            anchors = list(latest_stances.values())[-min(80, self.limit // 3):]
            anchor_markers = {
                (
                    item.get("timestamp"),
                    item.get("kind"),
                    item.get("topic_key"),
                    item.get("position"),
                )
                for item in anchors
            }
            recent_slots = max(1, self.limit - len(anchors))
            recent = records[-recent_slots:]
            merged = anchors + recent

            kept: list[dict[str, Any]] = []
            seen: set[tuple[Any, Any, Any, Any]] = set()
            for item in merged:
                marker = (
                    item.get("timestamp"),
                    item.get("kind"),
                    item.get("topic_key"),
                    item.get("position"),
                )
                if marker in seen:
                    continue
                seen.add(marker)
                kept.append(item)

            # If an anchor also occurred in the recent window, retain one copy.
            # Keep ordering stable enough for "latest record wins" semantics.
            kept.sort(key=lambda item: str(item.get("timestamp", "")))
            self.path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in kept[-self.limit:]),
                encoding="utf-8",
            )
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
            confidence = _clamp(raw.get("confidence", 0.5))
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

    def consider_stances(
        self,
        updates: Iterable[dict[str, Any]],
        *,
        generation: int,
        provider: str,
        basis_episode_ids: Iterable[str] = (),
        grounded_sources: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Accumulate proposed self-positions and promote only stable ones.

        Promotion requires:
        - support in at least two different generations,
        - at least one grounded source or observed experience,
        - confidence >= 0.55.

        Mature stances remain explicitly revisable. A later revision must itself
        mature before it replaces the currently established position.
        """
        records = self._read_records()
        basis = [str(value) for value in basis_episode_ids if str(value).strip()][:16]
        grounded_now = sorted({
            str(value).strip()[:240]
            for value in grounded_sources
            if str(value).strip()
        })[:12]
        out: list[dict[str, Any]] = []

        for raw in updates:
            if not isinstance(raw, dict):
                continue
            topic = str(raw.get("topic", "")).strip()[:500]
            position = str(raw.get("position", raw.get("statement", ""))).strip()[:3000]
            reasoning = str(raw.get("reasoning", "")).strip()[:3000]
            relation = str(raw.get("relation", "new")).strip().lower()
            if relation not in {"new", "reinforce", "revise", "challenge", "uncertain"}:
                relation = "new"
            if not topic or not position:
                continue

            confidence = _clamp(raw.get("confidence", 0.5))
            key = _topic_key(topic)
            related = [
                record
                for record in records
                if record.get("kind") in {"stance_candidate", "stance"}
                and record.get("topic_key") == key
            ]
            previous = related[-1] if related else None
            previous_position = str(previous.get("position", "")) if previous else ""
            similarity = _position_similarity(position, previous_position)
            compatible = (
                previous is not None
                and relation not in {"revise", "challenge"}
                and (relation == "reinforce" or similarity >= 0.34)
            )

            if compatible:
                generations = {
                    int(value)
                    for value in previous.get("support_generations", [])
                    if isinstance(value, int) or str(value).isdigit()
                }
                generations.add(int(generation))
                support_count = max(
                    int(previous.get("support_count", 1) or 1) + 1,
                    len(generations),
                )
                grounded = sorted(set(previous.get("grounded_sources", [])) | set(grounded_now))[:16]
            else:
                generations = {int(generation)}
                support_count = 1
                grounded = grounded_now

            mature = (
                confidence >= 0.55
                and len(generations) >= 2
                and support_count >= 2
                and bool(grounded)
            )
            kind = "stance" if mature else "stance_candidate"
            identity_weight = (
                min(0.85, 0.22 + 0.1 * len(generations) + 0.06 * len(grounded))
                if mature else 0.0
            )
            statement = (
                f"Current revisable position on {topic}: {position}"
                if mature
                else f"Candidate position on {topic}: {position}"
            )
            value = self._append({
                "generation": generation,
                "kind": kind,
                "topic": topic,
                "topic_key": key,
                "position": position,
                "statement": statement[:3500],
                "reasoning": reasoning,
                "relation": relation,
                "epistemic_status": "self_position" if mature else "proposed_self_position",
                "confidence": round(confidence, 4),
                "identity_weight": round(identity_weight, 4),
                "revisable": True,
                "support_count": support_count,
                "support_generations": sorted(generations)[-12:],
                "grounded_sources": grounded,
                "basis_episode_ids": basis,
                "provider": provider[:80],
                "similarity_to_previous": round(similarity, 4),
            })
            out.append(value)
            records.append(value)
        return out

    def current_stances(self, limit: int = 12) -> list[dict[str, Any]]:
        """Return the latest mature stance for each topic."""
        latest: dict[str, dict[str, Any]] = {}
        for record in self._read_records():
            if record.get("kind") != "stance":
                continue
            key = str(record.get("topic_key", "")).strip()
            if not key:
                continue
            latest[key] = record

        ranked = sorted(
            latest.values(),
            key=lambda record: (
                float(record.get("identity_weight", 0.0) or 0.0),
                int(record.get("generation", 0) or 0),
            ),
            reverse=True,
        )
        return ranked[: max(0, int(limit))]

    def context(self, recent_limit: int = 8, stance_limit: int = 10) -> list[dict[str, Any]]:
        """Persistent identity context: mature stances plus recent self history."""
        stances = self.current_stances(stance_limit)
        recent = self.recent(recent_limit)
        seen = {
            (
                item.get("timestamp"),
                item.get("kind"),
                item.get("topic_key"),
                item.get("statement"),
            )
            for item in stances
        }
        for item in recent:
            marker = (
                item.get("timestamp"),
                item.get("kind"),
                item.get("topic_key"),
                item.get("statement"),
            )
            if marker not in seen:
                stances.append(item)
                seen.add(marker)
        return stances[-(stance_limit + recent_limit):]

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        return self._read_records()[-max(0, limit):]
