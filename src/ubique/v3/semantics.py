'''Evidence-first local semantic learning from independently sourced passages.

This subsystem forms *retrievable, revisable* comparisons. It never converts
text co-occurrence into a scientific fact or a confident personal belief.
'''
from __future__ import annotations

import re
import random  # stochastic exploratory associations
from typing import Set

from .hippocampus import Hippocampus, terms
from .text_quality import clean_source, content_terms, meaningful_sentences
from .world_model import WorldModel, key
from .representation import Cortex
from .storage import Journal

BOILERPLATE = ("project gutenberg", "license", "copyright",
               "produced by", "ebook", "electronic edition", "transcriber")

GENERIC_TERMS = {
    "about", "after", "again", "against", "also", "another", "because",
    "being", "between", "could", "every", "first", "from", "have",
    "however", "into", "itself", "more", "most", "much", "must",
    "only", "other", "others", "over", "same", "some", "such",
    "than", "that", "their", "them", "there", "these", "those",
    "through", "under", "upon", "very", "were", "what", "when",
    "where", "which", "while", "with", "without", "would", "your",
    "many", "make", "made", "might", "still", "then", "this",
}


def meaningful_overlap(words) -> set[str]:
    return {word for word in words if len(word) >= 4 and word not in GENERIC_TERMS}


def source_sentence(text: str) -> str:
    chunks = meaningful_sentences(text, limit=8)
    for sentence in chunks:
        value = " ".join(sentence.split())
        if (65 <= len(value) <= 330 and len(terms(value)) >= 6
                and not any(t in value.casefold() for t in BOILERPLATE)):
            return value
    return ""


def evidence_excerpt(text: str) -> str:
    """Return an exact, readable source excerpt even for short passages."""
    quote = source_sentence(text)
    if quote:
        return quote
    alternatives = meaningful_sentences(text, limit=4)
    if alternatives:
        return alternatives[0][:480]
    return " ".join(clean_source(text).split())[:350]


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
            shared‑term threshold from 2 to 1 for a given observation.
        """
        self.journal = journal
        self.hippocampus = hippocampus
        self.world = world
        self.cortex = cortex
        self.comparisons = {str(ev["data"].get("id")) for ev in journal.events()
                            if ev["kind"] in {"insight","association_hypothesis"}}
        self.exploration_factor = exploration_factor
        self._term_frequency: dict[str, int] = {}
        self._seen_terms: Set[str] = self._build_seen_terms_cache()
        # Counter for consecutive observations that yield no new comparisons.
        self._no_comp_counter: int = 0

    def _build_seen_terms_cache(self) -> Set[str]:
        seen: Set[str] = set()
        for ev in self.hippocampus.all_episodes():
            txt = str(ev.get("text", ""))
            title = str(ev.get("payload", {}).get("title", ""))
            concepts = set(terms(txt, 24)) | set(terms(title, 12))
            seen.update(concepts)
            for word in concepts:
                self._term_frequency[word] = self._term_frequency.get(word, 0) + 1
        return seen

    @staticmethod
    def family(record: dict) -> str:
        source = str(record.get("source", ""))
        return source if source.startswith(("library:", "http://", "https://")) else ""

    # ---------------------------------------------------------------------
    # Adaptive exploration helpers
    # ---------------------------------------------------------------------
    def _adjust_exploration_factor(self, produced_comparisons: bool) -> None:
        """Adapt ``self.exploration_factor`` using a streak counter.

        When consecutive observations produce no new cross‑source comparisons,
        the learner becomes increasingly exploratory, raising the chance of
        relaxing the shared‑term threshold. A successful observation resets the
        streak and gently reduces exploration.
        """
        if produced_comparisons:
            # Reset the failure streak and slightly reduce exploration.
            self._no_comp_counter = 0
            self.exploration_factor = max(self.exploration_factor - 0.01, 0.05)
        else:
            # Increment the streak and increase exploration more aggressively.
            self._no_comp_counter = getattr(self, "_no_comp_counter", 0) + 1
            # Scale the increment: 0.02 per consecutive failure, capped at 0.10.
            increment = min(0.02 * self._no_comp_counter, 0.10)
            self.exploration_factor = min(self.exploration_factor + increment, 0.40)

    def observe(self, episode: dict, *, generation: int | None = None) -> list[dict]:
        if episode.get("epistemic_status") != "source_passage":
            return []
        eid = episode.get("id")
        if not eid or not self.family(episode):
            return []
        result: list[dict] = []
        sentence = source_sentence(str(episode.get("text", "")))
        if sentence:
            claim = self.world.claim(
                "Source passage: " + sentence, status="SOURCE_CLAIM",
                source_refs=[eid], confidence=0.28, generation=generation)
            result.append({"type": "quotation", "claim": claim["id"], "evidence_ids": [eid]})

        current = meaningful_overlap(content_terms(str(episode.get(
            "text", ""))[:1700], 36))
        current |= meaningful_overlap(content_terms(str(episode.get(
            "payload", {}).get("title", "")), 12))
        earlier = self.hippocampus.all_episodes()
        choices: list[tuple[float, dict, set[str]]] = []

        for other in reversed(earlier):
            if other.get("id") == eid or other.get("epistemic_status") != "source_passage":
                continue
            family = self.family(other)
            if not family or family == self.family(episode):
                continue
            concepts = meaningful_overlap(content_terms(str(other.get(
                "text", ""))[:1700], 36))
            concepts |= meaningful_overlap(content_terms(str(other.get(
                "payload", {}).get("title", "")), 12))
            shared = current & concepts
            
            min_shared = 2
            if random.random() < self.exploration_factor:
                min_shared = 1
            if len(shared) == 1:
                term = next(iter(shared))
                if term not in self._seen_terms:
                    min_shared = 1

            if len(shared) >= min_shared:
                # Enhanced rarity weighting: emphasize rare conceptual bridges
                novelty_bonus = sum(2.5 / (1 + self._term_frequency.get(t, 0)) for t in shared)
                relevance_score = float(len(shared)) + novelty_bonus
                choices.append((relevance_score, other, shared))
            if len(choices) >= 80:
                break

        if not choices:
            for other in reversed(earlier):
                if other.get("id") == eid or other.get("epistemic_status") != "source_passage":
                    continue
                family = self.family(other)
                if not family or family == self.family(episode):
                    continue
                concepts = meaningful_overlap(content_terms(str(other.get(
                    "text", ""))[:1700], 36))
                concepts |= meaningful_overlap(content_terms(str(other.get(
                    "payload", {}).get("title", "")), 12))
                shared = current & concepts
                if len(shared) >= 1:
                    relevance_score = float(len(shared)) + sum(2.5 / (1 + self._term_frequency.get(t, 0)) for t in shared)
                    choices.append((relevance_score, other, shared))
                if len(choices) >= 80:
                    break

        # Sort by computed relevance score (incorporating both term count and enhanced rarity bridge bonus)
        choices.sort(key=lambda row: row[0], reverse=True)

        for _, other, shared in choices[:2]:
            pair = sorted([eid, other["id"]])
            fingerprint = "comparison:" + key("|".join(pair))
            if fingerprint in self.comparisons:
                continue
            sorted_shared = sorted(shared)[:5]
            if not sorted_shared:
                continue
            comparison = {
                "id": fingerprint, "kind": "cross_source_comparison",
                "topic": ", ".join(sorted_shared),
                "source_families": [self.family(episode), self.family(other)],
                "evidence_ids": pair,
                "statement": ("Two independent source passages share conceptual "
                              "hinges in terms: " + ", ".join(sorted_shared) +
                              ". Their interpretations may illuminate or contrast; "
                              "this is an invitation to examine, not a definitive conclusion."),
                "uncertainty": 0.78}
            first = evidence_excerpt(str(episode.get("text", "")))
            second = evidence_excerpt(str(other.get("text", "")))
            comparison["passages"] = [first, second]
            comparison["question"] = (
                "In what ways do the perspectives of both authors intersect or diverge regarding " +
                 ", ".join(sorted_shared[:3]) + "? What further source evidence could test this relationship?")
            comparison["epistemic_status"] = "unverified_association"
            comparison["interpretation_quality"] = "exploratory_not_entailment"
            link = self.world.claim(comparison["statement"], status="HYPOTHESIS",
                                    source_refs=pair, confidence=0.10,
                                    generation=generation)
            comparison["claim_id"] = link["id"]
            self.journal.append("association_hypothesis", comparison, generation=generation)
            self.comparisons.add(fingerprint)
            result.append(comparison)
            left = self.cortex.ensure(sorted_shared[0])
            right = self.cortex.ensure(sorted_shared[-1])
            self.cortex.connect(left.id, right.id, relation="compared_in_sources",
                                weight=0.18, event_id=fingerprint)
        # -----------------------------------------------------------------
        # Adjust exploration factor based on whether we produced any new
        # cross‑source comparisons during this call.
        # -----------------------------------------------------------------
        self._adjust_exploration_factor(produced_comparisons=bool(result))
        latest = set(terms(str(episode.get("text", "")), 24))
        latest.update(terms(str(episode.get("payload", {}).get("title", "")), 12))
        for word in latest:
            self._seen_terms.add(word)
            self._term_frequency[word] = self._term_frequency.get(word, 0) + 1
        return result
