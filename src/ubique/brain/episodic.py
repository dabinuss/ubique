from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from ..memory import MEMORY_DIR


STOPWORDS = {
    "about", "after", "again", "also", "because", "been", "before", "being",
    "between", "could", "does", "from", "have", "into", "just", "more", "most",
    "only", "other", "should", "some", "than", "that", "their", "there", "these",
    "they", "this", "through", "very", "what", "when", "where", "which", "while",
    "with", "would", "your", "eine", "einen", "einer", "einem", "eines", "aber",
    "auch", "dass", "dies", "diese", "dieser", "durch", "haben", "kann", "mehr",
    "nicht", "oder", "sich", "sind", "über", "wird", "werden", "wurde", "wären",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w]+", " ", str(text).lower(), flags=re.UNICODE)
    return {
        token
        for token in cleaned.split()
        if len(token) >= 4 and token not in STOPWORDS
    }


def extract_concepts(text: str, limit: int = 10) -> list[str]:
    words = list(tokens(text))
    words.sort(key=lambda value: (-len(value), value))
    return words[: max(0, limit)]


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class EpisodeStore:
    """Append-only fast episodic memory with associative retrieval."""

    def __init__(self, path: Path | None = None, limit: int = 2500):
        self.path = path or (MEMORY_DIR / "brain_episodes.jsonl")
        self.limit = max(100, int(limit))

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
        return out

    def append(
        self,
        *,
        kind: str,
        text: str,
        source: str,
        concepts: Iterable[str] | None = None,
        epistemic_status: str = "observed",
        novelty: float = 0.5,
        surprise: float = 0.5,
        salience: float = 0.5,
        payload: dict[str, Any] | None = None,
        generation: int | None = None,
    ) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now()
        text = str(text)[:16000]
        concept_list = [
            str(value).strip().lower()[:120]
            for value in (concepts or extract_concepts(text))
            if str(value).strip()
        ]
        concept_list = list(dict.fromkeys(concept_list))[:24]
        digest = hashlib.sha1(
            f"{timestamp}|{kind}|{source}|{text[:1000]}".encode("utf-8")
        ).hexdigest()[:14]
        record = {
            "id": f"episode:{digest}",
            "timestamp": timestamp,
            "generation": generation,
            "kind": str(kind)[:80],
            "source": str(source)[:160],
            "epistemic_status": str(epistemic_status)[:80],
            "text": text,
            "concepts": concept_list,
            "novelty": round(clamp(novelty), 4),
            "surprise": round(clamp(surprise), 4),
            "salience": round(clamp(salience), 4),
            "payload": payload if isinstance(payload, dict) else {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trim()
        return record

    def _trim(self) -> None:
        records = self._read_all()
        if len(records) <= self.limit:
            return
        old = records[:-self.limit]
        anchors = sorted(
            old,
            key=lambda item: (
                float(item.get("salience", 0.0)),
                float(item.get("surprise", 0.0)),
                float(item.get("novelty", 0.0)),
            ),
            reverse=True,
        )[: min(50, max(0, self.limit // 10))]
        recent_slots = max(1, self.limit - len(anchors))
        recent = records[-recent_slots:]
        merged = anchors + recent
        seen: set[str] = set()
        kept = []
        for item in merged:
            episode_id = str(item.get("id", ""))
            if not episode_id or episode_id in seen:
                continue
            seen.add(episode_id)
            kept.append(item)
        kept = kept[-self.limit:]
        self.path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in kept),
            encoding="utf-8",
        )

    def recent(self, limit: int = 12) -> list[dict[str, Any]]:
        return self._read_all()[-max(0, limit):]

    @staticmethod
    def _similarity(query_tokens: set[str], query_concepts: set[str], episode: dict[str, Any]) -> float:
        episode_tokens = tokens(str(episode.get("text", "")))
        episode_concepts = {
            str(value).lower()
            for value in episode.get("concepts", [])
            if str(value).strip()
        }
        token_union = query_tokens | episode_tokens
        concept_union = query_concepts | episode_concepts
        token_jaccard = (
            len(query_tokens & episode_tokens) / len(token_union)
            if token_union else 0.0
        )
        concept_jaccard = (
            len(query_concepts & episode_concepts) / len(concept_union)
            if concept_union else 0.0
        )
        return 0.58 * concept_jaccard + 0.42 * token_jaccard

    def recall(
        self,
        query: str = "",
        *,
        concepts: Iterable[str] | None = None,
        limit: int = 6,
        include_imagined: bool = True,
    ) -> list[dict[str, Any]]:
        records = self._read_all()
        if not records:
            return []
        query_tokens = tokens(query)
        query_concepts = {
            str(value).strip().lower()
            for value in (concepts or [])
            if str(value).strip()
        }
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, dict[str, Any]]] = []
        for index, episode in enumerate(records):
            if (
                not include_imagined
                and str(episode.get("epistemic_status", "")).lower() == "imagined"
            ):
                continue
            similarity = self._similarity(query_tokens, query_concepts, episode)
            dt = _parse_time(episode.get("timestamp"))
            if dt is not None:
                age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
                recency = 1.0 / (1.0 + age_hours / 48.0)
            else:
                recency = (index + 1) / max(1, len(records))
            salience = clamp(episode.get("salience", 0.0))
            surprise = clamp(episode.get("surprise", 0.0))
            score = 0.68 * similarity + 0.14 * salience + 0.09 * surprise + 0.09 * recency

            # Memory provenance affects recall strength. Observed outcomes and
            # external sources can dominate when relevant; model proposals are
            # useful but weaker, and imagined counterfactuals are deliberately
            # faint unless no grounded alternative exists.
            epistemic_status = str(episode.get("epistemic_status", "")).lower()
            provenance_factor = {
                "imagined": 0.38,
                "model_proposal": 0.72,
                "model_hypothesis": 0.72,
                "model_interpretation": 0.78,
            }.get(epistemic_status, 1.0)
            score *= provenance_factor

            if query_tokens or query_concepts:
                if similarity <= 0.0 and score < 0.18:
                    continue
            scored.append((score, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        out = []
        for score, episode in scored[: max(0, limit)]:
            copy = dict(episode)
            copy["recall_score"] = round(score, 4)
            out.append(copy)
        return out

    def novelty_against_memory(
        self,
        text: str,
        concepts: Iterable[str] | None = None,
    ) -> float:
        matches = self.recall(text, concepts=concepts, limit=1)
        if not matches:
            return 1.0
        episode = matches[0]
        similarity = self._similarity(
            tokens(text),
            {str(value).lower() for value in (concepts or [])},
            episode,
        )
        return clamp(1.0 - similarity)

    def replay_candidates(self, limit: int = 8) -> list[dict[str, Any]]:
        records = self._read_all()
        if not records:
            return []
        ranked = sorted(
            records,
            key=lambda item: (
                0.42 * clamp(item.get("salience", 0.0))
                + 0.28 * clamp(item.get("surprise", 0.0))
                + 0.2 * clamp(item.get("novelty", 0.0))
                + 0.1 * (1.0 if item in records[-20:] else 0.0)
            ),
            reverse=True,
        )
        return ranked[: max(0, limit)]

    def count(self) -> int:
        return len(self._read_all())
