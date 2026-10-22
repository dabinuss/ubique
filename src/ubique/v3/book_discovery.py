"""Autonomous discovery of verifiably retrievable public-domain books."""
from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..library import (
    _load_index, _write_index, _normalized_title, _unique_id,
    _remote_source_url, utc_now,
)

CATALOGUE = "https://gutendex.com/books/"


def _text_url(formats: dict) -> str:
    for mime, url in formats.items():
        if mime.startswith("text/plain") and isinstance(url, str):
            if _remote_source_url({"source_url": url}):
                return url
    return ""


def discover_public_books(query: str, *, limit: int = 2) -> dict:
    """Find real public-domain sources and save catalogue entries.

    Titles, source URLs and provenance come from the catalogue; no
    reading or authorship is invented. Actual text is fetched only if
    the agent later chooses to read this book.
    """
    query = str(query).strip()[:160]
    if not query:
        return {"query": "", "added": [], "status": "empty_query"}
    request = Request(
        CATALOGUE + "?" + urlencode({"search": query}),
        headers={"User-Agent": "UbiqueResearch/3.0 (autonomous learning)"})
    with urlopen(request, timeout=12) as response:
        catalog = json.loads(response.read(250_000).decode("utf-8"))

    index = _load_index()
    items = index.setdefault("items", [])
    known = {_normalized_title(str(item.get("title", ""))) for item in items
             if item.get("status") in {"available", "offered"}}
    added = []
    for book in catalog.get("results", [])[:20]:
        if len(added) >= max(1, min(int(limit), 5)):
            break
        if book.get("copyright") is not False:
            continue
        title = str(book.get("title", "")).strip()[:300]
        url = _text_url(book.get("formats") or {})
        normalized = _normalized_title(title)
        if not title or not normalized or not url or normalized in known:
            continue
        query_terms = {x for x in _normalized_title(query).split()
                       if len(x) > 3 and x not in {"book", "human", "about", "with"}}
        if query_terms and not (query_terms & set(normalized.split())):
            continue
        wish = next((item for item in items
                     if item.get("status") == "wanted"
                     and _normalized_title(str(item.get("title", ""))) == normalized), None)
        if wish is None:
            wish = {
                "id": _unique_id(index, title), "title": title,
                "added_by": "ubique", "added_at": utc_now(),
                "read_count": 0, "read_cursor": 0, "fully_read": False,
                "fully_read_at": None, "notes": [], "content_path": "",
            }
            items.append(wish)
        wish.update({
            "kind": "book", "status": "available",
            "source": "Project Gutenberg (Gutendex catalogue)",
            "source_url": url, "license": "Public domain",
            "public_domain_catalogue_id": book.get("id"),
            "reason": "Autonomously discovered for: " + query,
        })
        known.add(normalized)
        added.append({"id": wish["id"], "title": title, "source_url": url})
    if added:
        _write_index(index)
    return {"query": query, "added": added, "status": "found" if added else "no_match"}
