"""Revisable source inquiries with speculative bridges clearly labelled."""
from __future__ import annotations
from .text_quality import meaningful_sentences, concept_terms
from .world_model import key


def compare_sources(left: dict, right: dict) -> dict | None:
    if left.get("source") == right.get("source"):
        return None
    la = meaningful_sentences(str(left.get("text", "")), 6)
    lb = meaningful_sentences(str(right.get("text", "")), 6)
    if not la or not lb:
        return None
    scored = []
    for a in la:
        aw = set(concept_terms(a, 35))
        for b in lb:
            bw = set(concept_terms(b, 35))
            shared = aw & bw
            if len(shared) >= 2:
                scored.append((len(shared), a, b, sorted(shared)))
    if scored:
        _, a, b, shared = max(scored, key=lambda row: row[0])
        quality = "source_linked"
        topic = ", ".join(shared[:3])
    else:
        a, b, shared = la[0], lb[0], []
        quality = "speculative"
        topic = "the relationship between these ideas"
    left_terms = set(concept_terms(a, 35))
    right_terms = set(concept_terms(b, 35))
    unique_left = sorted(left_terms - right_terms)[:2]
    unique_right = sorted(right_terms - left_terms)[:2]
    if quality == "speculative":
        first = ", ".join(unique_left) or "the first passage"
        second = ", ".join(unique_right) or "the second passage"
        question = (
            "The passages respectively discuss " + first + " and " + second +
            ". Is there any meaningful relationship between these ideas at all? "
            "What new evidence could support or reject one?"
        )
    elif unique_left and unique_right:
        question = (
            "How do the authors approach " + topic + " differently, given their "
            "emphases on " + ", ".join(unique_left) + " and " +
            ", ".join(unique_right) + "? Which assumptions differ?"
        )
    else:
        question = (
            "How do both authors approach " + topic +
            " in their separate passages? Which assumptions differ?"
        )
    return {
        "question": question, "passages": [a[:480], b[:480]],
        "topics": shared[:6], "source_ids": [left["id"], right["id"]],
        "distinctive_terms": [unique_left, unique_right],
        "status": "open_question", "bridge_quality": quality,
        "uncertainty": .95 if quality == "speculative" else .8,
    }


def enquire(hippocampus, journal, world, *, generation: int | None = None) -> dict | None:
    books = [
        e for e in reversed(hippocampus.all_episodes())
        if e.get("epistemic_status") == "source_passage"
        and str(e.get("source", "")).startswith("library:")
    ]
    if len(books) < 2:
        return None
    already = {str(e.get("data", {}).get("id", ""))
               for e in journal.events() if e.get("kind") == "inquiry"}
    speculative = None
    selected = None
    for a in books[:12]:
        for b in books[:36]:
            if a.get("source") == b.get("source") or a.get("id") == b.get("id"):
                continue
            inquiry = compare_sources(a, b)
            if inquiry is None:
                continue
            identifier = "inquiry:" + key("|".join(sorted(inquiry["source_ids"])))
            if identifier in already:
                continue
            inquiry["id"] = identifier
            if inquiry["bridge_quality"] == "source_linked":
                selected = inquiry
                break
            if speculative is None:
                speculative = inquiry
        if selected is not None:
            break
    if selected is None:
        if generation is None or generation % 17 != 0:
            return None
        selected = speculative
    if selected is None:
        return None
    journal.append("inquiry", selected, generation=generation)
    world.claim(
        "Open question: " + selected["question"],
        status="HYPOTHESIS", source_refs=selected["source_ids"],
        confidence=.08 if selected["bridge_quality"] == "speculative" else .15,
        generation=generation,
    )
    return selected
