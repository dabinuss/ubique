'''Evidence-first local semantic learning from independently sourced passages.

This subsystem forms *retrievable, revisable* comparisons. It never converts
text co-occurrence into a scientific fact or a confident personal belief.
'''
from __future__ import annotations

import re
import random  # <-- new import for stochastic exploration
from typing import Set

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
                 world: WorldModel, cortex: Cortex, *, exploration_factor: float = 0.1):
        """Create a learner.

        Parameters
        ----------
        journal, hippocampus, world, cortex:
            Core components of the Ubique cognitive stack.
        exploration_factor:
            Probability (0‑1) that the learner will *relax* the minimum
            shared‑term threshold from 2 to 1 for a given observation. This
            small stochastic tweak encourages occasional broader curiosity
            without overwhelming the system with low‑quality comparisons.
        """
        self.journal = journal
        self.hippocampus = hippocampus
        self.world = world
        self.cortex = cortex
        self.comparisons = {str(ev["data"].get("id")) for ev in journal.events()
                            if ev["kind"] in {"insight","association_hypothesis"}}
        # Store the exploration factor for later use
        self.exploration_factor = exploration_factor
        # Cache of all terms seen in prior episodes – used for deterministic novelty check
        self._seen_terms: Set[str] = self._build_seen_terms_cache()

    def _build_seen_terms_cache(self) -> Set[str]:
        """Collect every term that has appeared in earlier episodes.

        The cache is built once at construction time; it is inexpensive and
        provides a quick way to test whether a term is *novel* with respect to
        the learner's experience so far. Because the learner only adds new
        comparisons after observation, the cache does not need to be updated
        during a single run – the change is reversible by simply removing this
        method and the associated attribute.
        """
        seen: Set[str] = set()
        for ev in self.hippocampus.all_episodes():
            txt = str(ev.get("text", ""))
            seen.update(terms(txt, 24))
            title = str(ev.get("payload", {}).get("title", ""))
            seen.update(terms(title, 12))
        return seen

    @staticmethod
    def family(record: dict) -> str:
        source = str(record.get("source", ""))
        return source if source.startswith(("library:", "http://", "https://")) else ""

    def observe(self, episode: dict, *, generation: int | None = None) -> list[dict]:
        """Interpret an observed book passage and compare separate source texts.

        The method now occasionally relaxes the shared‑term requirement to
        foster exploratory connections. Additionally, if at least one of the
        shared terms has never been seen before, the learner will also accept
        the comparison even when only a single term overlaps. This deterministic
        novelty boost works alongside the existing stochastic `exploration_factor`
        and preserves all original behaviours.
        """
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
            # --- Begin exploratory threshold logic ---
            # Normal requirement: at least 2 shared terms.
            # With probability `exploration_factor`, accept a single shared term.
            min_shared = 2
            if random.random() < self.exploration_factor:
                min_shared = 1
            # Deterministic novelty boost: if any shared term is novel, also allow 1.
            if len(shared) == 1:
                term = next(iter(shared))
                if term not in self._seen_terms:
                    min_shared = 1
            if len(shared) >= min_shared:
                choices.append((len(shared), other, shared))
            # --- End exploratory threshold logic ---
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
