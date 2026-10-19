"""Separate source prose from public-domain boilerplate without policing ideas.

This is an attention aid, not a ban-list for subjects. Original excerpts remain
in episodic memory; cleaning only influences semantic sampling and activation.
"""
from __future__ import annotations
import re

# Explicit publishing metadata, not topics or philosophical positions.
FRONT_MATTER = (
    "project gutenberg", "gutenberg ebook", "gutenberg license",
    "produced by", "transcribed by", "electronic edition",
    "copyright notice", "www.gutenberg.org", "gutenberg.org",
    "you may copy", "you may redistribute", "terms of use",
    "the full project gutenberg", "start of the project gutenberg",
    "end of the project gutenberg", "most people start at our website",
    "anyone anywhere", "united states and most other",
)
META_TERMS = {
    "gutenberg", "ebook", "ebooks", "license", "licence", "copyright",
    "transcriber", "proofreading", "distributed", "redistribute",
    "anyone", "anywhere", "www", "http", "https",
}
START = re.compile(r"\\*{3}\\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\\n]*\\*{3}", re.I)
END = re.compile(r"\\*{3}\\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)

def clean_source(text: str) -> str:
    """Preserve author text while excluding obvious Gutenberg headers/licenses.

    A short literary mention of Gutenberg is preserved; only recognizable
    publishing lines or long boilerplate fragments are omitted.
    """
    text = str(text or "")
    match = START.search(text)
    if match:
        text = text[match.end():]
    match = END.search(text)
    if match:
        text = text[:match.start()]
    lines = []
    skipping = False
    for line in text.splitlines():
        value = " ".join(line.lower().split())
        if "the full project gutenberg license" in value:
            skipping = True
        if skipping:
            continue
        if any(marker in value for marker in FRONT_MATTER):
            continue
        if value.startswith(("release date:", "language:", "character set encoding:",
                             "title:", "author:", "credits:", "updated:")):
            continue
        lines.append(line)
    return "\\n".join(lines)

def content_terms(text: str, limit: int = 24) -> list[str]:
    from .hippocampus import terms
    # Filtering stays intentionally narrow: grammar, ideology and unexpected
    # subjects are never globally rejected.
    return [term for term in terms(clean_source(text), max(64, limit * 5))
            if term not in META_TERMS][:limit]

def meaningful_sentences(text: str, limit: int = 10) -> list[str]:
    clean = clean_source(text)
    chunks = re.split(r"(?<=[.!?])\\s+", " ".join(clean.split()))
    choices = []
    for chunk in chunks:
        sentence = chunk.strip()
        if not 50 <= len(sentence) <= 650:
            continue
        if len(content_terms(sentence, 24)) < 5:
            continue
        if any(marker in sentence.casefold() for marker in FRONT_MATTER):
            continue
        choices.append(sentence)
        if len(choices) >= limit:
            break
    return choices
