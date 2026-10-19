"""Locally generated, revisable questions and cross-source interpretations.

Not a semantic entailment engine: preserve the actual sentences and their
provenance, and make disagreements/questions explorable without hard gates.
"""
from __future__ import annotations

from .text_quality import meaningful_sentences, content_terms
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
        aw = set(content_terms(a, 35))
        for b in lb:
            bw = set(content_terms(b, 35))
            shared = aw & bw
            # This is an exploratory overlap, not an entailment test.
            if len(shared) >= 2:
                scored.append((len(shared), a, b, sorted(shared)))
    if not scored:
        # If the books cover different subjects, curiosity can still ask how
        # one relates to the other; don't invent agreement.
        a, b = la[0], lb[0]
        shared = []
    else:
        _, a, b, shared = max(scored, key=lambda row: row[0])
    topic = ", ".join(shared[:3]) or "the relationship between these ideas"
    question = ("How does the view in the first passage relate to the second "
                "concerning " + topic + "? Which assumptions differ?")
    return {"question": question, "passages": [a[:480], b[:480]],
            "topics": shared[:6], "source_ids": [left["id"], right["id"]],
            "status": "open_question", "uncertainty": 0.8}

def enquire(hippocampus, journal, world, *, generation: int | None = None) -> dict | None:
    books = [e for e in reversed(hippocampus.all_episodes())
             if e.get("epistemic_status") == "source_passage"
             and str(e.get("source", "")).startswith("library:")]
    if len(books) < 2:
        return None
    already = {str(ev.get("data", {}).get("id", "")) for ev in journal.events()
               if ev.get("kind") == "inquiry"}
    # Recent passages are preferred, but anything in living memory can be revisited.
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
            journal.append("inquiry", inquiry, generation=generation)
            # A question does not become a factual claim merely by being asked.
            world.claim("Open question: " + inquiry["question"], status="HYPOTHESIS",
                        source_refs=inquiry["source_ids"], confidence=0.15,
                        generation=generation)
            return inquiry
    return None
