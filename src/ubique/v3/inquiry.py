"""Revisable source inquiries, with speculation distinguished from evidence.

Unexpected conceptual leaps are welcome, but a lack of shared content must
never be described as an observed connection or an established disagreement.
"""
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
        status = "source_linked"
        topic = ", ".join(shared[:3])
    else:
        # A free association remains possible, but is *not* evidence of a link.
        a, b = la[0], lb[0]
        shared = []
        status = "speculative"
        topic = "a possible relationship between these ideas"

    left_terms = set(concept_terms(a, 35))
    right_terms = set(concept_terms(b, 35))
    unique_left = sorted(left_terms - right_terms)[:2]
    unique_right = sorted(right_terms - left_terms)[:2]

    if status == "speculative":
        left_cue = ", ".join(unique_left) or "the first passage"
        right_cue = ", ".join(unique_right) or "the second passage"
        question = (
            "The passages respectively discuss " + left_cue + " and " +
            right_cue + ". Is there any meaningful relationship between " +
            "these ideas at all? What new evidence could support or reject one?"
        )
    elif unique_left and unique_right:
        question = (
            "How do the authors connect or distinguish " + topic +
            ", given their different emphases on " + ", ".join(unique_left) +
            " and " + ", ".join(unique_right) +
            "? What specific passage might challenge this comparison?"
        )
    else:
        question = (
            "How do both authors approach " + topic +
            "? Which distinct assertions support or challenge a comparison?"
        )
    return {
        "question": question,
        "passages": [a[:480], b[:480]],
        "topics": shared[:6],
        "source_ids": [left["id"], right["id"]],
        "distinctive_terms": [unique_left, unique_right],
        "status": "open_question",
        "bridge_quality": status,
        "uncertainty": 0.95 if status == "speculative" else 0.8,
    }


def enquire(hippocampus, journal, world, *, generation: int | None = None) -> dict | None:
    books = [
        e for e in reversed(hippocampus.all_episodes())
        if e.get("epistemic_status") == "source_passage"
        and str(e.get("source", "")).startswith("library:")
    ]
    if len(books) < 2:
        return None
    already = {
        str(ev.get("data", {}).get("id", ""))
        for ev in journal.events() if ev.get("kind") == "inquiry"
    }
    speculative = None
    # Try genuinely overlapping authors first; speculative bridges remain
    # accessible occasionally instead of dominating every new heartbeat.
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
        else:
            continue
        break
    else:
        selected = None
    if selected is None:
        # Creative connections are invited, but only periodically when
        # there is no evidence-linked pair to investigate.
        if generation is None or generation % 17 != 0:
            return None
        selected = speculative
    if selected is None:
        return None
    journal.append("inquiry", selected, generation=generation)
    world.claim(
        "Open question: " + selected["question"],
        status="HYPOTHESIS", source_refs=selected["source_ids"],
        confidence=0.08 if selected["bridge_quality"] == "speculative" else 0.15,
        generation=generation,
    )
    return selected
