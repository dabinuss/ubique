import json

import ubique.library as library


def _redirect(tmp_path, monkeypatch):
    root = tmp_path / "library"
    monkeypatch.setattr(library, "LIBRARY_DIR", root)
    monkeypatch.setattr(library, "ITEMS_DIR", root / "items")
    monkeypatch.setattr(library, "INDEX_PATH", root / "index.json")


def test_library_reads_full_text_only_after_explicit_selection(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    added = library.add_library_item("A book", "abcdefghij" * 200, source="offered by user", actor="user")
    item_id = added["item"]["id"]
    catalog = library.library_catalog()
    assert catalog[0]["title"] == "A book"
    assert "excerpt" not in catalog[0]
    first = library.read_library_item(item_id, offset=0, max_chars=500)
    assert len(first["excerpt"]) == 500
    assert first["next_offset"] == 500


def test_ubique_can_request_book_without_claiming_to_have_read_it(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    result = library.apply_library_action({
        "action": "request",
        "title": "Interesting book",
        "source": "author/title reference",
        "reason": "I want to compare its account of identity with my current view.",
    })
    assert result["action"] == "request"
    assert result["item"]["status"] == "wanted"
    read = library.read_library_item(result["item"]["id"])
    assert read["content_available"] is False
    assert read["excerpt"] == ""


def test_library_notes_persist_without_becoming_instructions(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    item = library.add_library_item("Text", "hello")["item"]
    library.add_library_note(item["id"], "This challenges my earlier view.")
    data = json.loads(library.INDEX_PATH.read_text(encoding="utf-8"))
    assert data["items"][0]["notes"][0]["text"].startswith("This challenges")
