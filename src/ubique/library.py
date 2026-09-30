from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .memory import MEMORY_DIR
from .state import utc_now


LIBRARY_DIR = MEMORY_DIR / "library"
ITEMS_DIR = LIBRARY_DIR / "items"
INDEX_PATH = LIBRARY_DIR / "index.json"
VALID_ACTIONS = {"list", "read", "add", "request", "note"}
MAX_STORED_TEXT_CHARS = 200_000
MAX_READ_CHARS = 12_000


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


def library_catalog(limit: int = 30) -> list[dict[str, Any]]:
    """Return metadata only. Library contents are never injected automatically."""
    index = _load_index()
    out: list[dict[str, Any]] = []
    for item in index.get("items", [])[: max(0, limit)]:
        if not isinstance(item, dict):
            continue
        out.append({
            "id": str(item.get("id", "")),
            "title": str(item.get("title", "")),
            "kind": str(item.get("kind", "text")),
            "status": str(item.get("status", "available")),
            "source": str(item.get("source", "")),
            "content_available": bool(item.get("content_path")),
            "added_by": str(item.get("added_by", "unknown")),
            "read_count": int(item.get("read_count", 0) or 0),
        })
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


def read_library_item(item_id: str, offset: int = 0, max_chars: int = 8000) -> dict[str, Any]:
    index = _load_index()
    item = _find_item(index, item_id)
    if item is None:
        raise ValueError(f"unknown library item: {item_id}")

    offset = max(0, int(offset or 0))
    max_chars = max(500, min(int(max_chars or 8000), MAX_READ_CHARS))
    path = _content_path(item)
    text = ""
    if path is not None and path.exists():
        text = path.read_text(encoding="utf-8")

    excerpt = text[offset: offset + max_chars]
    next_offset = offset + len(excerpt)
    item["read_count"] = int(item.get("read_count", 0) or 0) + 1
    item["last_selected_at"] = utc_now()
    item["selected_by"] = "ubique"
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
        "next_offset": next_offset if next_offset < len(text) else None,
        "total_chars": len(text),
        "notes": list(item.get("notes", []))[-8:],
    }


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
        "notes": [],
    }
    index.setdefault("items", []).append(item)
    _write_index(index)
    return {"action": "add", "item": item}


def request_library_item(title: str, source: str = "", reason: str = "", actor: str = "ubique") -> dict[str, Any]:
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
    if action == "read":
        return read_library_item(
            str(spec.get("item_id", "")),
            int(spec.get("offset", 0) or 0),
            int(spec.get("max_chars", 8000) or 8000),
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
