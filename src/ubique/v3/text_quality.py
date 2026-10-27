"""Separate genuine source prose from publisher boilerplate and dialogue glue.

Nothing here bans a subject, source, question or speculative interpretation.
The original passages remain available for free association and rereading.
"""
from __future__ import annotations
import re

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
# These are poor standalone comparison topics, although the original
# passages may of course use them meaningfully in their normal contexts.
DISCOURSE_TERMS = {
    "every", "other", "others", "another", "also", "therefore",
    "however", "although", "either", "neither", "rather", "whereas",
    "which", "whose", "while", "after", "before", "there", "these",
    "those", "here", "upon", "allow", "allowed", "because", "being",
    "having", "shall", "might", "could", "would", "should", "given",
    "following", "further",
}
CATALOGUE_TERMS = {
    "illustrated", "catalogue", "catalog", "ebook", "volume", "vol",
    "printed", "published", "edition", "index", "contents", "pages",
}
START = re.compile(r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}", re.I)
END = re.compile(r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def clean_source(text: str) -> str:
    """Remove recognizable publishing lines, never the author's original ideas."""
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
    return "\n".join(lines)


def content_terms(text: str, limit: int = 24) -> list[str]:
    from .hippocampus import terms
    return [term for term in terms(clean_source(text), max(64, limit * 5))
            if term not in META_TERMS][:limit]


def concept_terms(text: str, limit: int = 24) -> list[str]:
    """Substantive attention cues; not a truth or curiosity restriction."""
    stop = DISCOURSE_TERMS | CATALOGUE_TERMS
    return [term for term in content_terms(text, max(80, limit * 6))
            if term not in stop][:limit]


def meaningful_sentences(text: str, limit: int = 10) -> list[str]:
    clean = clean_source(text)
    chunks = re.split(r"(?<=[.!?])\s+", " ".join(clean.split()))
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
