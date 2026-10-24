"""Discover genuine, readable public-domain books without inventing titles.

Search is curiosity-led. Precise research wishes are *not* treated as book
titles: when a narrow query misses, look for relevant related works and record
which topic led to the discovery. A catalogue listing is not a reading event.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..library import (
    _load_index, _write_index, _normalized_title, _unique_id,
    _remote_source_url, utc_now,
)

CATALOGUE = "https://gutendex.com/books/"
STOP = {"the", "and", "for", "with", "from", "between", "about", "analysis",
        "study", "studies", "measurement", "measurements", "quantitative",
        "empirical", "direct", "review", "systematic", "simultaneous"}
RELATED = {
    "crying": ("emotion", "psychology"),
    "lacrimal": ("emotion", "physiology"),
    "emg": ("physiology", "psychology"),
    "autonomic": ("physiology", "nervous"),
    "neural": ("brain", "psychology"),
    "neurobiology": ("brain", "biology"),
    "consciousness": ("mind", "psychology"),
    "learning": ("education", "psychology"),
}


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", text.casefold())
            if word not in STOP}


def _text_url(formats: dict) -> str:
    for mime, url in (formats or {}).items():
        if mime.startswith("text/plain") and isinstance(url, str):
            if _remote_source_url({"source_url": url}):
                return url
    return ""


def _queries(query: str) -> list[str]:
    """At most three bounded requests, first literal and then less specific."""
    words = [word for word in re.findall(r"[a-z]{4,}", query.casefold())
             if word not in STOP]
    options = [query]
    if len(words) > 1:
        options.append(" ".join(words[:2]))
    # A narrow experimental-paper title is unlikely in a historical book
    # catalogue. Explore a nearby subject, not a fabricated matching title.
    for word in words:
        if word in RELATED:
            options.extend(RELATED[word])
            break
    options.extend(words[:2])
    return list(dict.fromkeys(x for x in options if x))[:3]


def _relevance(book: dict, query: str, searched: str) -> int:
    title = str(book.get("title", ""))
    authors = " ".join(str(a.get("name", "")) for a in book.get("authors", [])
                       if isinstance(a, dict))
    subjects = " ".join(str(s) for s in book.get("subjects", []))
    book_words = _words(title + " " + authors + " " + subjects)
    requested = _words(query)
    related = {synonym for word in requested for synonym in RELATED.get(word, ())}
    matching = requested & book_words
    # Do not accept every book returned by a broad catalogue search.
    # The derived subject may supply a bridge, but never a claim of identity.
    bridges = related & book_words
    return 4 * len(_words(title) & requested) + 2 * len(matching) + len(bridges)


def discover_public_books(query: str, *, limit: int = 2) -> dict:
    """Add traceable, previously unknown works selected by their relevance."""
    query = str(query).strip()[:160]
    if not query:
        return {"query": "", "added": [], "status": "empty_query"}
    index = _load_index()
    items = index.setdefault("items", [])
    known = {_normalized_title(str(item.get("title", ""))) for item in items
             if item.get("status") in {"available", "offered"}}
    found: dict[str, tuple[int, dict, str, str]] = {}
    failures = []
    for search in _queries(query):
        request = Request(CATALOGUE + "?" + urlencode({"search": search}),
                          headers={"User-Agent": "UbiqueResearch/3.0 (autonomous learning)"})
        try:
            with urlopen(request, timeout=12) as response:
                catalogue = json.loads(response.read(250_000).decode("utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            failures.append(type(exc).__name__)
            continue
        for book in catalogue.get("results", [])[:30]:
            if not isinstance(book, dict) or book.get("copyright") is not False:
                continue
            title = str(book.get("title", "")).strip()[:300]
            url = _text_url(book.get("formats") or {})
            normalized = _normalized_title(title)
            if not normalized or normalized in known or not url:
                continue
            score = _relevance(book, query, search)
            if score <= 0:
                continue
            if normalized not in found or score > found[normalized][0]:
                found[normalized] = (score, book, url, search)
        if len(found) >= max(1, int(limit)):
            break
    added = []
    for _, book, url, search in sorted(found.values(),
                                     key=lambda record: record[0], reverse=True):
        if len(added) >= max(1, min(int(limit), 5)):
            break
        title = str(book["title"]).strip()[:300]
        normalized = _normalized_title(title)
        if normalized in known:
            continue
        wish = next((item for item in items
                     if item.get("status") == "wanted"
                     and _normalized_title(str(item.get("title", ""))) == normalized), None)
        if wish is None:
            wish = {"id": _unique_id(index, title), "title": title,
                    "added_by": "ubique", "added_at": utc_now(),
                    "read_count": 0, "read_cursor": 0, "fully_read": False,
                    "fully_read_at": None, "notes": [], "content_path": ""}
            items.append(wish)
        wish.update({
            "kind": "book", "status": "available",
            "source": "Project Gutenberg (Gutendex catalogue)",
            "source_url": url, "license": "Public domain",
            "public_domain_catalogue_id": book.get("id"),
            "reason": "Autonomously discovered while investigating: " + query,
            "discovery_query": search,
        })
        known.add(normalized)
        added.append({"id": wish["id"], "title": title, "source_url": url})
    if added:
        _write_index(index)
    return {"query": query, "added": added,
            "status": "found" if added else ("source_unavailable" if failures else "no_match"),
            "attempted_queries": _queries(query), "errors": failures}
