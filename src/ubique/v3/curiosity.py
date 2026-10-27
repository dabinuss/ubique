"""Curiosity with revisable interests and meaningful attention selection."""
from __future__ import annotations
import hashlib
import re
import time
from typing import Iterable
from .storage import Journal
from .text_quality import DISCOURSE_TERMS, CATALOGUE_TERMS, META_TERMS


def normalized(text: str) -> set[str]:
    stop = {"of", "the", "and", "between", "human", "during", "direct",
            "measurements", "quantitative", "empirical", "an", "in", "with",
            "on", "a", "to", "for", "study", "review", "der", "die", "das",
            "und", "von", "eine", "einer", "zum"}
    return {w for w in re.findall(r"[a-zäöüß]{3,}", text.casefold())
            if w not in stop}


def related(a: str, b: str) -> float:
    wa, wb = normalized(a), normalized(b)
    return len(wa & wb) / max(1, len(wa | wb))


def hashlib_score(text: str, generation: int) -> int:
    return int(hashlib.sha256(
        (str(generation // 6) + text).encode()).hexdigest()[:8], 16)


class Curiosity:
    def __init__(self, journal: Journal):
        self.journal = journal
        self.interests = {}
        self.wishes = []
        for ev in journal.events():
            if ev["kind"] == "interest":
                d = ev["data"]
                self.interests[d["topic"]] = d
            if ev["kind"] == "wish":
                self.wishes.append(ev["data"])

    def notice(self, topic: str, *, novelty=.4, progress=.0,
               uncertainty=.5, source="", generation=None):
        topic = " ".join(topic.split())[:160]
        if not topic:
            return {}
        previous = self.interests.get(topic, {})
        visits = int(previous.get("visits", 0)) + 1
        saturation = max(0.0, .09 * (visits - 2))
        record = {
            "topic": topic, "visits": visits, "novelty": float(novelty),
            "progress": float(progress), "uncertainty": float(uncertainty),
            "saturation": min(1, saturation),
            "interest": max(0, min(1, .30 * novelty + .35 * max(0, progress)
                        + .35 * uncertainty - saturation
                        + float(previous.get("interest", 0)) * .25)),
            "source": source, "last_seen": time.time(),
        }
        self.journal.append("interest", record, generation=generation)
        self.interests[topic] = record
        return record

    def score(self, topic: str, *, bridge=.0) -> float:
        known = self.interests.get(topic, {})
        base = known.get("interest", .15) + .25 * bridge - .35 * known.get("saturation", 0)
        last = known.get("last_seen", 0)
        boost = min(max(0, time.time() - last) / 86400 / 5, 1) * .1 if last else 0
        return base + boost

    def select(self, candidates: Iterable[str], *, generation=0) -> str:
        options = list(dict.fromkeys(x for x in candidates if x))
        if not options:
            return ""
        weak = DISCOURSE_TERMS | CATALOGUE_TERMS | META_TERMS
        def score(topic):
            label = str(topic).casefold().strip()
            metadata_penalty = -.35 if label in weak else 0
            title_penalty = -.45 if len(label) > 110 else 0
            diversity = .04 * ((hashlib_score(topic, generation) % 13) / 13)
            return self.score(topic) + metadata_penalty + title_penalty + diversity
        return max(options, key=score)

    def similar_wish(self, title: str, threshold=.50) -> dict | None:
        for wish in self.wishes:
            if related(title, str(wish.get("title", ""))) >= threshold:
                return wish
        return None

    def wish(self, title: str, *, reason="", generation=None) -> dict:
        match = self.similar_wish(title)
        if match:
            return {**match, "already_requested": True}
        item = {"title": title[:250], "reason": reason[:500],
                "already_requested": False}
        self.journal.append("wish", item, generation=generation)
        self.wishes.append(item)
        return item
