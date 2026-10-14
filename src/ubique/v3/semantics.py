"""Evidence-first local semantic learning from independently sourced passages.

This subsystem forms *retrievable, revisable* comparisons. It never converts
text co-occurrence into a scientific fact or a confident personal belief.
"""
from __future__ import annotations

import re
from .hippocampus import Hippocampus, terms
from .world_model import WorldModel, key
from .representation import Cortex
from .storage import Journal

BOILERPLATE = ("project gutenberg", "license", "copyright",
               "produced by", "ebook", "electronic edition", "transcriber")

def source_sentence(text: str) -> str:
    chunks = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    for sentence in chunks:
        value = " ".join(sentence.split())
        if (65 <= len(value) <= 330 and len(terms(value)) >= 6
                and not any(t in value.casefold() for t in BOILERPLATE)):
            return value
    return ""

class SemanticLearner:
    def __init__(self, journal: Journal, hippocampus: Hippocampus,
                 world: WorldModel, cortex: Cortex):
        self.journal = journal
        self.hippocampus = hippocampus
        self.world = world
        self.cortex = cortex
        self.comparisons = {str(ev["data"].get("id")) for ev in journal.events()
                            if ev["kind"] in {"insight","association_hypothesis"}}

    @staticmethod
    def family(record: dict) -> str:
        source = str(record.get("source", ""))
        return source if source.startswith(("library:", "http://", "https://")) else ""

    def observe(self, episode: dict, *, generation: int | None = None) -> list[dict]:
        """Interpret an observed book passage and compare separate source texts."""
        if episode.get("epistemic_status") != "source_passage":
            return []
        eid = episode.get("id")
        if not eid or not self.family(episode):
            return []
        result = []
        sentence = source_sentence(str(episode.get("text", "")))
        if sentence:
            claim = self.world.claim(
                "Source passage: " + sentence, status="SOURCE_CLAIM",
                source_refs=[eid], confidence=0.28, generation=generation)
            result.append({"type": "quotation", "claim": claim["id"], "evidence_ids": [eid]})

        current = set(terms(str(episode.get("text", ""))[:1700], 24))
        current |= set(terms(str(episode.get("payload", {}).get("title", "")), 12))
        earlier = self.hippocampus.all_episodes()
        choices: list[tuple[int, dict, set[str]]] = []
        for other in reversed(earlier):
            if other.get("id") == eid or other.get("epistemic_status") != "source_passage":
                continue
            family = self.family(other)
            if not family or family == self.family(episode):
                continue
            concepts = set(terms(str(other.get("text", ""))[:1700], 24))
            concepts |= set(terms(str(other.get("payload", {}).get("title", "")), 12))
            shared = current & concepts
            if len(shared) >= 2:
                choices.append((len(shared), other, shared))
            if len(choices) >= 80:
                break
        choices.sort(key=lambda row: row[0], reverse=True)
        for _, other, shared in choices[:2]:
            pair = sorted([eid, other["id"]])
            fingerprint = "comparison:" + key("|".join(pair))
            if fingerprint in self.comparisons:
                continue
            shared = sorted(shared)[:5]
            if not shared:
                continue
            comparison = {
                "id": fingerprint, "kind": "cross_source_comparison",
                "topic": ", ".join(shared),
                "source_families": [self.family(episode), self.family(other)],
                "evidence_ids": pair,
                "statement": ("Two independent source passages contain overlapping "
                              "terms: " + ", ".join(shared) +
                              ". Their interpretations may agree or disagree; "
                              "this is an invitation to investigate, not proof."),
                "uncertainty": 0.75}
            # Word overlap is useful for asking a question, not a confirmed
            # insight or evidence for the authors agreeing on the same claim.
            comparison["epistemic_status"] = "unverified_association"
            self.journal.append("association_hypothesis", comparison, generation=generation)
            self.comparisons.add(fingerprint)
            link = self.world.claim(comparison["statement"], status="HYPOTHESIS",
                                    source_refs=pair, confidence=0.12,
                                    generation=generation)
            comparison["claim_id"] = link["id"]
            result.append(comparison)
            left = self.cortex.ensure(shared[0])
            right = self.cortex.ensure(shared[-1])
            self.cortex.connect(left.id, right.id, relation="compared_in_sources",
                                weight=0.16, event_id=fingerprint)
        return result
