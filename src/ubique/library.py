from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .memory import MEMORY_DIR
from .state import utc_now


LIBRARY_DIR = MEMORY_DIR / "library"
ITEMS_DIR = LIBRARY_DIR / "items"
INDEX_PATH = LIBRARY_DIR / "index.json"
VALID_ACTIONS = {"list", "read", "add", "request", "note", "sync"}
MAX_STORED_TEXT_CHARS = 200_000
MAX_READ_CHARS = 12_000
MAX_REMOTE_TEXT_BYTES = 8_000_000
REMOTE_TIMEOUT_SECONDS = 20
ALLOWED_REMOTE_HOSTS = {"www.gutenberg.org", "gutenberg.org"}


def _empty_index() -> dict[str, Any]:
    return {"version": 1, "items": []}


def _load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return _empty_index()
    try:
        value = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty_index()
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        return _empty_index()
    return value


def _write_index(value: dict[str, Any]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(INDEX_PATH)


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (value or "item")[:64]


def _normalized_title(title: str) -> str:
    """Compare reading wishes by meaningfully normalized title, not slug suffix."""
    return re.sub(r"[\W_]+", " ", str(title).casefold(), flags=re.UNICODE).strip()


def _unique_id(index: dict[str, Any], title: str) -> str:
    existing = {str(item.get("id", "")) for item in index.get("items", []) if isinstance(item, dict)}
    base = _slug(title)
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def _find_item(index: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    item_id = item_id.strip()
    for item in index.get("items", []):
        if isinstance(item, dict) and str(item.get("id", "")) == item_id:
            return item
    return None


def library_catalog(
    limit: int = 30,
    *,
    include_completed: bool = True,
) -> list[dict[str, Any]]:
    """Return metadata only. Library contents are never injected automatically."""
    index = _load_index()
    limit = max(0, int(limit))
    if limit == 0:
        return []
    out: list[dict[str, Any]] = []
    for item in index.get("items", []):
        if not isinstance(item, dict):
            continue
        if not include_completed and bool(item.get("fully_read", False)):
            continue
        out.append({
            "id": str(item.get("id", "")),
            "title": str(item.get("title", "")),
            "author": str(item.get("author", "")),
            "kind": str(item.get("kind", "text")),
            "status": str(item.get("status", "available")),
            "source": str(item.get("source", "")),
            "source_url": str(item.get("source_url", "")),
            "license": str(item.get("license", "")),
            "topics": list(item.get("topics", []))[:8] if isinstance(item.get("topics"), list) else [],
            "content_available": bool(item.get("content_path") or item.get("source_url")),
            "added_by": str(item.get("added_by", "unknown")),
            "read_count": int(item.get("read_count", 0) or 0),
            "read_cursor": max(0, int(item.get("read_cursor", 0) or 0)),
            "fully_read": bool(item.get("fully_read", False)),
            "fully_read_at": item.get("fully_read_at"),
        })
        if len(out) >= limit:
            break
    return out


def library_catalog_prompt(limit: int = 20) -> str:
    catalog = library_catalog(limit)
    if not catalog:
        return "- empty"
    return json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))


def _content_path(item: dict[str, Any]) -> Path | None:
    rel = str(item.get("content_path", "")).strip().replace("\\", "/")
    if not rel:
        return None
    path = (LIBRARY_DIR / rel).resolve()
    root = LIBRARY_DIR.resolve()
    if path != root and root not in path.parents:
        return None
    return path


def _remote_source_url(item: dict[str, Any]) -> str:
    url = str(item.get("source_url", "")).strip()
    if not url:
        return ""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_REMOTE_HOSTS:
        return ""
    return url


def _load_item_text(index: dict[str, Any], item: dict[str, Any]) -> str:
    """Load local text or fetch an allowlisted public-domain source once.

    Remote books are cached into the persistent library on first explicit read,
    so later reads do not depend on the network and the heartbeat can commit the
    cached text with the rest of memory.
    """
    path = _content_path(item)
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8")

    source_url = _remote_source_url(item)
    if not source_url:
        return ""

    request = Request(
        source_url,
        headers={"User-Agent": "Ubique/0.1 public-domain library reader"},
    )
    with urlopen(request, timeout=REMOTE_TIMEOUT_SECONDS) as response:
        data = response.read(MAX_REMOTE_TEXT_BYTES + 1)
        if len(data) > MAX_REMOTE_TEXT_BYTES:
            raise ValueError(
                f"remote library text exceeds {MAX_REMOTE_TEXT_BYTES} bytes"
            )
        charset = response.headers.get_content_charset() or "utf-8"

    text = data.decode(charset, errors="replace")
    item_id = str(item.get("id", "")).strip()
    if text and item_id:
        ITEMS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = ITEMS_DIR / f"{_slug(item_id)}.txt"
        cache_path.write_text(text, encoding="utf-8")
        item["content_path"] = f"items/{cache_path.name}"
        item["cached_from"] = source_url
        item["cached_at"] = utc_now()
        _write_index(index)
    return text


def sync_library_items(
    item_ids: list[str] | None = None,
    *,
    limit: int = 30,
) -> dict[str, Any]:
    """Cache remote library items without marking them as read."""
    index = _load_index()
    wanted = {
        str(value).strip()
        for value in (item_ids or [])
        if str(value).strip()
    }
    limit = max(0, min(int(limit or 30), 100))
    results: list[dict[str, Any]] = []

    for item in index.get("items", []):
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()
        if wanted and item_id not in wanted:
            continue
        if not item_id or not _remote_source_url(item):
            continue

        path = _content_path(item)
        if path is not None and path.exists():
            results.append({
                "id": item_id,
                "status": "already_cached",
                "chars": len(path.read_text(encoding="utf-8")),
            })
            continue

        try:
            text = _load_item_text(index, item)
            results.append({
                "id": item_id,
                "status": "cached" if text else "unavailable",
                "chars": len(text),
            })
        except Exception as exc:
            results.append({
                "id": item_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })

    return {
        "action": "sync",
        "requested": sorted(wanted),
        "items": results,
        "cached": sum(1 for item in results if item["status"] in {"cached", "already_cached"}),
        "failed": sum(1 for item in results if item["status"] == "failed"),
    }


def read_library_item(
    item_id: str,
    offset: int | None = None,
    max_chars: int = 8000,
    *,
    reread: bool = False,
) -> dict[str, Any]:
    index = _load_index()
    item = _find_item(index, item_id)
    if item is None:
        raise ValueError(f"unknown library item: {item_id}")

    max_chars = max(500, min(int(max_chars or 8000), MAX_READ_CHARS))
    text = _load_item_text(index, item)

    stored_cursor = max(0, int(item.get("read_cursor", 0) or 0))
    fully_read = bool(item.get("fully_read", False))

    # A wish without an actual text is not a reading experience. Preserve
    # the existing progress and counter instead of recording a phantom read.
    if not text:
        return {
            "action": "read",
            "item": {
                "id": item.get("id"),
                "title": item.get("title"),
                "kind": item.get("kind", "text"),
                "status": item.get("status", "available"),
                "source": item.get("source", ""),
            },
            "offset": stored_cursor if offset is None else max(0, int(offset)),
            "excerpt": "",
            "content_available": False,
            "next_offset": None,
            "total_chars": 0,
            "read_cursor": stored_cursor,
            "fully_read": fully_read,
            "already_complete": False,
            "notes": list(item.get("notes", []))[-8:],
        }

    if offset is None:
        if reread:
            offset = 0
        elif fully_read:
            return {
                "action": "read",
                "item": {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "kind": item.get("kind", "text"),
                    "status": item.get("status", "available"),
                    "source": item.get("source", ""),
                },
                "offset": stored_cursor,
                "excerpt": "",
                "content_available": bool(text),
                "next_offset": None,
                "total_chars": len(text),
                "read_cursor": stored_cursor,
                "fully_read": True,
                "already_complete": True,
                "notes": list(item.get("notes", []))[-8:],
            }
        else:
            offset = stored_cursor
    offset = max(0, int(offset or 0))

    excerpt = text[offset: offset + max_chars]
    end_offset = offset + len(excerpt)
    next_offset = end_offset if end_offset < len(text) else None

    item["read_count"] = int(item.get("read_count", 0) or 0) + 1
    item["last_selected_at"] = utc_now()
    item["selected_by"] = "ubique"

    if text and (offset == stored_cursor or reread):
        item["read_cursor"] = max(stored_cursor, end_offset)
    elif "read_cursor" not in item:
        item["read_cursor"] = stored_cursor

    if text and int(item.get("read_cursor", 0) or 0) >= len(text):
        item["read_cursor"] = len(text)
        item["fully_read"] = True
        item["fully_read_at"] = utc_now()
    else:
        item["fully_read"] = False

    _write_index(index)

    return {
        "action": "read",
        "item": {
            "id": item.get("id"),
            "title": item.get("title"),
            "kind": item.get("kind", "text"),
            "status": item.get("status", "available"),
            "source": item.get("source", ""),
        },
        "offset": offset,
        "excerpt": excerpt,
        "content_available": bool(text),
        "next_offset": next_offset,
        "total_chars": len(text),
        "read_cursor": int(item.get("read_cursor", 0) or 0),
        "fully_read": bool(item.get("fully_read", False)),
        "already_complete": False,
        "notes": list(item.get("notes", []))[-8:],
    }



def mark_library_item_complete(item_id: str, *, completed_at: str | None = None) -> bool:
    """Persist completion inferred from trusted historical read outcomes."""
    index = _load_index()
    item = _find_item(index, item_id)
    if item is None:
        return False
    path = _content_path(item)
    if path is None or not path.exists():
        return False
    try:
        total_chars = len(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return False
    if total_chars <= 0:
        return False

    changed = (
        not bool(item.get("fully_read", False))
        or int(item.get("read_cursor", 0) or 0) != total_chars
    )
    item["read_cursor"] = total_chars
    item["fully_read"] = True
    item["fully_read_at"] = completed_at or item.get("fully_read_at") or utc_now()
    if changed:
        _write_index(index)
    return changed


def add_library_item(
    title: str,
    text: str = "",
    source: str = "",
    reason: str = "",
    kind: str = "text",
    actor: str = "ubique",
    status: str = "available",
) -> dict[str, Any]:
    title = title.strip()[:300]
    if not title:
        raise ValueError("library item requires a title")
    text = str(text or "")
    if len(text) > MAX_STORED_TEXT_CHARS:
        raise ValueError(f"library text exceeds {MAX_STORED_TEXT_CHARS} characters")

    index = _load_index()
    item_id = _unique_id(index, title)
    content_path = ""
    if text:
        ITEMS_DIR.mkdir(parents=True, exist_ok=True)
        path = ITEMS_DIR / f"{item_id}.md"
        path.write_text(text, encoding="utf-8")
        content_path = f"items/{path.name}"

    item = {
        "id": item_id,
        "title": title,
        "kind": (kind.strip() or "text")[:80],
        "status": status[:40],
        "source": source.strip()[:1000],
        "reason": reason.strip()[:1500],
        "added_by": actor[:80],
        "added_at": utc_now(),
        "content_path": content_path,
        "read_count": 0,
        "read_cursor": 0,
        "fully_read": False,
        "fully_read_at": None,
        "notes": [],
    }
    index.setdefault("items", []).append(item)
    _write_index(index)
    return {"action": "add", "item": item}


def request_library_item(title: str, source: str = "", reason: str = "", actor: str = "ubique") -> dict[str, Any]:
    # Repeated interest must not create a new book every cycle. This is only
    # idempotency of an already-open wish, not a restriction on what may be read.
    normalized = _normalized_title(title)
    if normalized:
        for item in _load_index().get("items", []):
            if (
                isinstance(item, dict)
                and item.get("kind") == "reading_request"
                and item.get("status") == "wanted"
                and _normalized_title(str(item.get("title", ""))) == normalized
            ):
                return {"action": "request", "item": item, "already_requested": True}

    result = add_library_item(
        title=title,
        text="",
        source=source,
        reason=reason,
        kind="reading_request",
        actor=actor,
        status="wanted",
    )
    result["action"] = "request"
    result["already_requested"] = False
    return result


def add_library_note(item_id: str, note: str, actor: str = "ubique") -> dict[str, Any]:
    note = note.strip()[:6000]
    if not note:
        raise ValueError("library note must not be empty")
    index = _load_index()
    item = _find_item(index, item_id)
    if item is None:
        raise ValueError(f"unknown library item: {item_id}")
    notes = item.setdefault("notes", [])
    notes.append({"at": utc_now(), "by": actor[:80], "text": note})
    item["notes"] = notes[-40:]
    _write_index(index)
    return {"action": "note", "item_id": item_id, "note": item["notes"][-1]}


def apply_library_action(spec: dict[str, Any], actor: str = "ubique") -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("library action must be a JSON object")
    action = str(spec.get("action", "list")).strip().lower()
    if action not in VALID_ACTIONS:
        raise ValueError(f"unknown library action: {action}")
    if action == "list":
        return {"action": "list", "items": library_catalog(int(spec.get("limit", 30) or 30))}
    if action == "sync":
        raw_ids = spec.get("item_ids", [])
        item_ids = raw_ids if isinstance(raw_ids, list) else []
        return sync_library_items(
            [str(value) for value in item_ids],
            limit=int(spec.get("limit", 30) or 30),
        )
    if action == "read":
        raw_offset = spec.get("offset")
        offset = None if raw_offset is None else int(raw_offset or 0)
        return read_library_item(
            str(spec.get("item_id", "")),
            offset,
            int(spec.get("max_chars", 8000) or 8000),
            reread=bool(spec.get("reread", False)),
        )
    if action == "request":
        return request_library_item(
            str(spec.get("title", "")),
            str(spec.get("source", "")),
            str(spec.get("reason", "")),
            actor,
        )
    if action == "note":
        return add_library_note(str(spec.get("item_id", "")), str(spec.get("text", "")), actor)
    return add_library_item(
        title=str(spec.get("title", "")),
        text=str(spec.get("text", "")),
        source=str(spec.get("source", "")),
        reason=str(spec.get("reason", "")),
        kind=str(spec.get("kind", "text")),
        actor=actor,
        status="available" if spec.get("text") else "reference",
    )
