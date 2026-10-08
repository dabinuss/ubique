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


def test_library_default_read_advances_cursor_and_stops_at_completion(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    item = library.add_library_item("Cursor book", "x" * 900, actor="user")["item"]

    first = library.read_library_item(item["id"], max_chars=500)
    assert first["offset"] == 0
    assert first["next_offset"] == 500
    assert first["read_cursor"] == 500
    assert first["fully_read"] is False

    second = library.read_library_item(item["id"], max_chars=500)
    assert second["offset"] == 500
    assert second["next_offset"] is None
    assert second["read_cursor"] == 900
    assert second["fully_read"] is True

    catalog = library.library_catalog()
    assert catalog[0]["fully_read"] is True
    assert catalog[0]["read_cursor"] == 900

    third = library.read_library_item(item["id"])
    assert third["already_complete"] is True
    assert third["excerpt"] == ""


def test_library_explicit_reread_is_possible_after_completion(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    item = library.add_library_item("Reread book", "abcdef" * 100, actor="user")["item"]
    completed = library.read_library_item(item["id"], max_chars=1000)
    assert completed["fully_read"] is True

    reread = library.read_library_item(item["id"], max_chars=500, reread=True)
    assert reread["offset"] == 0
    assert reread["excerpt"]
    assert reread["fully_read"] is True


def test_mark_library_item_complete_migrates_legacy_progress(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    item = library.add_library_item("Legacy book", "legacy content", actor="user")["item"]

    changed = library.mark_library_item_complete(
        item["id"],
        completed_at="2026-10-02T00:00:00+00:00",
    )

    assert changed is True
    catalog = library.library_catalog()
    assert catalog[0]["fully_read"] is True
    assert catalog[0]["read_cursor"] == len("legacy content")
    assert catalog[0]["fully_read_at"] == "2026-10-02T00:00:00+00:00"


def test_catalog_can_hide_completed_books_from_spontaneous_attention(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    item = library.add_library_item("Finished", "done", actor="user")["item"]
    library.read_library_item(item["id"], max_chars=500)

    assert library.library_catalog(include_completed=False) == []
    assert library.library_catalog(include_completed=True)[0]["fully_read"] is True


def test_remote_public_domain_book_is_readable_and_cached(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    library.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    library.INDEX_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "items": [
                    {
                        "id": "remote-book",
                        "title": "Remote book",
                        "author": "Example Author",
                        "kind": "book",
                        "status": "available",
                        "source": "Project Gutenberg eBook test",
                        "source_url": "https://www.gutenberg.org/cache/epub/1/pg1.txt",
                        "license": "Public domain",
                        "topics": ["mind"],
                        "added_by": "test",
                        "added_at": "2026-10-03T00:00:00+00:00",
                        "content_path": "",
                        "read_count": 0,
                        "read_cursor": 0,
                        "fully_read": False,
                        "fully_read_at": None,
                        "notes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Headers:
        @staticmethod
        def get_content_charset():
            return "utf-8"

    class Response:
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read(_limit):
            return ("consciousness and brain " * 100).encode("utf-8")

    monkeypatch.setattr(library, "urlopen", lambda *_args, **_kwargs: Response())

    catalog = library.library_catalog()
    assert catalog[0]["content_available"] is True
    assert catalog[0]["source_url"].startswith("https://www.gutenberg.org/")

    result = library.read_library_item("remote-book", max_chars=500)
    assert result["excerpt"].startswith("consciousness and brain")
    assert result["content_available"] is True

    saved = json.loads(library.INDEX_PATH.read_text(encoding="utf-8"))
    rel = saved["items"][0]["content_path"]
    assert rel
    assert (library.LIBRARY_DIR / rel).exists()


def test_sync_caches_remote_book_without_advancing_read_state(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    library.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    library.INDEX_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "items": [
                    {
                        "id": "sync-book",
                        "title": "Sync book",
                        "author": "Example Author",
                        "kind": "book",
                        "status": "available",
                        "source": "Project Gutenberg eBook test",
                        "source_url": "https://www.gutenberg.org/ebooks/1.txt.utf-8",
                        "license": "Public domain",
                        "topics": ["mind"],
                        "added_by": "test",
                        "added_at": "2026-10-03T00:00:00+00:00",
                        "content_path": "",
                        "read_count": 0,
                        "read_cursor": 0,
                        "fully_read": False,
                        "fully_read_at": None,
                        "notes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Headers:
        @staticmethod
        def get_content_charset():
            return "utf-8"

    class Response:
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read(_limit):
            return b"brain mind consciousness"

    monkeypatch.setattr(library, "urlopen", lambda *_args, **_kwargs: Response())

    result = library.apply_library_action({"action": "sync", "limit": 10})
    assert result["cached"] == 1
    saved = json.loads(library.INDEX_PATH.read_text(encoding="utf-8"))
    item = saved["items"][0]
    assert item["content_path"]
    assert item["read_count"] == 0
    assert item["read_cursor"] == 0
    assert item["fully_read"] is False


def test_repeat_reading_wish_is_idempotent_across_case_and_punctuation(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    first = library.request_library_item(
        "Physiology of emotional crying",
        reason="Investigate emotional expression.",
    )
    repeated = library.request_library_item(
        "PHYSIOLOGY--OF--EMOTIONAL CRYING",
        reason="Investigate tear physiology in detail.",
    )

    assert first["already_requested"] is False
    assert repeated["already_requested"] is True
    assert repeated["item"]["id"] == first["item"]["id"]
    saved = json.loads(library.INDEX_PATH.read_text(encoding="utf-8"))
    assert len(saved["items"]) == 1
    assert saved["items"][0]["read_count"] == 0


def test_missing_book_does_not_count_as_reading(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    item = library.request_library_item("Unfulfilled research wish")["item"]
    for _ in range(3):
        result = library.read_library_item(item["id"])
        assert result["content_available"] is False
        assert result["excerpt"] == ""
        assert result["already_complete"] is False
        assert result["read_cursor"] == 0

    stored = json.loads(library.INDEX_PATH.read_text(encoding="utf-8"))["items"][0]
    assert stored["read_count"] == 0
    assert stored["read_cursor"] == 0
    assert stored["fully_read"] is False
