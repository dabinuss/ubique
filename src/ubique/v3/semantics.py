"""Source-grounded, revisable comparisons. Association never implies truth."""
from __future__ import annotations
import random
from .hippocampus import terms
from .text_quality import clean_source, concept_terms, meaningful_sentences
from .world_model import key

GENERIC_TERMS = {
    "about", "after", "again", "against", "also", "another", "because",
    "being", "between", "could", "every", "first", "from", "have",
    "however", "into", "itself", "more", "most", "much", "must", "only",
    "other", "others", "over", "same", "some", "such", "than", "that",
    "their", "them", "there", "these", "those", "through", "under",
    "upon", "very", "were", "what", "when", "where", "which", "while",
    "with", "without", "would", "your", "many", "make", "made",
    "might", "still", "then", "this",
}
BOILERPLATE = ("project gutenberg", "license", "copyright", "produced by",
               "ebook", "electronic edition", "transcriber")


def meaningful_overlap(words) -> set[str]:
    return {w for w in words if len(w) >= 3 and w not in GENERIC_TERMS}


def source_sentence(text: str) -> str:
    for sentence in meaningful_sentences(text, limit=8):
        s = " ".join(sentence.split())
        if (65 <= len(s) <= 330 and len(terms(s)) >= 6
                and not any(t in s.casefold() for t in BOILERPLATE)):
            return s
    return ""


def evidence_excerpt(text: str) -> str:
    quote = source_sentence(text)
    if quote:
        return quote
    choices = meaningful_sentences(text, limit=4)
    return choices[0][:480] if choices else " ".join(clean_source(text).split())[:350]


class SemanticLearner:
    def __init__(self, journal, hippocampus, world, cortex, *, exploration_factor=.1):
        self.journal, self.hippocampus = journal, hippocampus
        self.world, self.cortex = world, cortex
        self.exploration_factor = exploration_factor
        self.comparisons = {
            str(e["data"].get("id")) for e in journal.events()
            if e["kind"] in {"insight", "association_hypothesis"}
        }
        self._term_frequency = {}
        self._seen_terms = self._build_seen_terms_cache()
        self._no_comp_counter = 0

    @staticmethod
    def family(record: dict) -> str:
        src = str(record.get("source", ""))
        return src if src.startswith(("library:", "http://", "https://")) else ""

    def _concepts(self, record):
        text = str(record.get("text", ""))[:1700]
        cues = meaningful_overlap(concept_terms(text, 36))
        title = str(record.get("payload", {}).get("title", ""))
        if len(title) <= 120:
            cues.update(meaningful_overlap(concept_terms(title, 12)))
        return cues

    def _build_seen_terms_cache(self):
        seen = set()
        for ep in self.hippocampus.all_episodes():
            concepts = self._concepts(ep)
            seen.update(concepts)
            for term in concepts:
                self._term_frequency[term] = self._term_frequency.get(term, 0) + 1
        return seen

    def _adjust_exploration_factor(self, produced_comparisons: bool):
        if produced_comparisons:
            self._no_comp_counter = 0
            self.exploration_factor = max(.05, self.exploration_factor - .01)
        else:
            self._no_comp_counter += 1
            self.exploration_factor = min(
                .40, self.exploration_factor + min(.02 * self._no_comp_counter, .10))

    def observe(self, episode: dict, *, generation=None) -> list[dict]:
        status = episode.get("epistemic_status")
        if status not in {"source_passage", "exploratory_question", "association_hypothesis", "insight"}:
            return []
        eid = episode.get("id")
        if not eid or not self.family(episode):
            return []
        result = []
        sentence = source_sentence(str(episode.get("text", "")))
        if sentence:
            claim = self.world.claim(
                "Source passage: " + sentence, status="SOURCE_CLAIM",
                source_refs=[eid], confidence=.28, generation=generation)
            result.append({"type": "quotation", "claim": claim["id"],
                           "evidence_ids": [eid]})
        current = self._concepts(episode)
        earlier = self.hippocampus.all_episodes()
        choices = []
        for other in reversed(earlier):
            if (other.get("id") == eid or
                    other.get("epistemic_status") not in {"source_passage", "exploratory_question", "association_hypothesis", "insight"}):
                continue
            # Relax strict family restriction if options are extremely scarce
            if self.family(other) and self.family(other) == self.family(episode) and len(earlier) > 10:
                continue
            shared = current & self._concepts(other)
            min_shared = 1 if random.random() < self.exploration_factor else 2
            if len(shared) == 1 and self._term_frequency.get(next(iter(shared)), 0) <= 2:
                min_shared = 1
            if len(shared) < min_shared:
                continue
            novelty = sum(2.5 / (1 + self._term_frequency.get(t, 0)) for t in shared)
            choices.append((len(shared) + novelty, other, shared))
            if len(choices) >= 80:
                break
        # Do not manufacture an empty-topic comparison: earlier code added
        # such a choice only to silently discard it a few lines later.
        choices.sort(key=lambda entry: entry[0], reverse=True)
        produced = 0
        if not choices and len(current) >= 2:
            for other in reversed(earlier):
                if (other.get("id") == eid or
                        other.get("epistemic_status") != "source_passage" or
                        not self.family(other) or
                        self.family(other) == self.family(episode)):
                    continue
                other_concepts = self._concepts(other)
                if not other_concepts:
                    continue
                rare_shared = {t for t in (current | other_concepts) if self._term_frequency.get(t, 0) <= 1}
                if rare_shared:
                    choices.append((1.5, other, rare_shared))
                    if len(choices) >= 10:
                        break
            if not choices:
                # Adaptive semantic bridging: pair with the most recent diverse source using its top general concepts
                for other in reversed(earlier):
                    if (other.get("id") == eid or
                            other.get("epistemic_status") != "source_passage" or
                            not self.family(other) or
                            self.family(other) == self.family(episode)):
                        continue
                    other_concepts = self._concepts(other)
                    if other_concepts:
                        bridge_terms = list(current | other_concepts)[:4]
                        if bridge_terms:
                            choices.append((1.0, other, set(bridge_terms)))
                            break
            choices.sort(key=lambda entry: entry[0], reverse=True)

        for _, other, shared in choices[:2]:
            pair = sorted([eid, other["id"]])
            identifier = "comparison:" + key("|".join(pair))
            if identifier in self.comparisons or not shared:
                continue
            words = sorted(shared)[:5]
            question = ("Where do these authors agree or differ regarding " +
                        ", ".join(words[:3]) + "? What further evidence could test this?")
            comparison = {
                "id": identifier, "kind": "cross_source_comparison",
                "topic": ", ".join(words), "source_families": [
                    self.family(episode), self.family(other)],
                "evidence_ids": pair,
                "statement": ("Two passages share conceptual terms: " +
                              ", ".join(words) +
                              ". This is a hypothesis for exploration, not proof."),
                "uncertainty": .78, "passages": [
                    evidence_excerpt(str(episode.get("text", ""))),
                    evidence_excerpt(str(other.get("text", "")))],
                "question": question, "epistemic_status": "unverified_association",
                "interpretation_quality": "exploratory_not_entailment",
            }
            claim = self.world.claim(
                comparison["statement"], status="HYPOTHESIS",
                source_refs=pair, confidence=.10, generation=generation)
            comparison["claim_id"] = claim["id"]
            self.journal.append("association_hypothesis", comparison,
                                generation=generation)
            self.comparisons.add(identifier)
            result.append(comparison)
            produced += 1
            a, b = self.cortex.ensure(words[0]), self.cortex.ensure(words[-1])
            self.cortex.connect(a.id, b.id, relation="compared_in_sources",
                                weight=.18, event_id=identifier)
        # An extracted quotation by itself is not a successful comparison.
        self._adjust_exploration_factor(produced_comparisons=produced > 0)

        # Fallback: if no cross‑source comparison was produced but we have a
        # meaningful sentence, generate a self‑directed exploratory question.
        if produced == 0 and sentence:
            cues = meaningful_overlap(concept_terms(sentence, 12))
            if cues:
                words = sorted(cues)[:3]
                # Augment curiosity by suggesting additional high-frequency concepts
                frequent_terms = sorted(
                    (t for t in self._term_frequency if t not in cues),
                    key=lambda t: self._term_frequency[t],
                    reverse=True)[:3]
                extra = ", ".join(frequent_terms)
                if extra:
                    question = ("What insights can be drawn from this passage about " +
                                ", ".join(words) + "? Also consider " + extra + ".")
                else:
                    question = ("What insights can be drawn from this passage about " +
                                ", ".join(words) + "?")
                exploration = {
                    "type": "exploratory_question",
                    "question": question,
                    "evidence_ids": [eid],
                    "epistemic_status": "unverified_association",
                }
                # Record in journal for later reflection.
                self.journal.append("exploratory_question", exploration,
                                    generation=generation)
                result.append(exploration)

        for term in self._concepts(episode):
            self._seen_terms.add(term)
            self._term_frequency[term] = self._term_frequency.get(term, 0) + 1
        return result
