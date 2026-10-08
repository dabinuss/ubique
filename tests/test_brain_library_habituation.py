import json

import ubique.brain.runtime as runtime_module
from ubique.brain.action_selection import ActionCandidate
from ubique.brain.episodic import EpisodeStore
from ubique.brain.runtime import NeurocognitiveRuntime


def _runtime(tmp_path):
    runtime = NeurocognitiveRuntime.__new__(NeurocognitiveRuntime)
    runtime.episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    return runtime


def test_legacy_complete_read_is_inferred_from_action_outcome(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    progress = {"fully_read": False}

    def fake_catalog(limit=20, include_completed=True):
        if progress["fully_read"] and not include_completed:
            return []
        return [{
            "id": "book",
            "title": "Book",
            "fully_read": progress["fully_read"],
        }]

    monkeypatch.setattr(runtime_module, "library_catalog", fake_catalog)
    migrated = []

    def mark_complete(item_id, completed_at=None):
        progress["fully_read"] = True
        migrated.append((item_id, completed_at))
        return True

    monkeypatch.setattr(runtime_module, "mark_library_item_complete", mark_complete)
    runtime.episodes.append(
        kind="internal_action_outcome",
        text=json.dumps({
            "action": "read",
            "item": {"id": "book"},
            "content_available": True,
            "next_offset": None,
            "total_chars": 1000,
        }),
        source="action:library",
        epistemic_status="observed_action_outcome",
    )

    assert runtime._completed_library_items() == {"book"}
    assert migrated and migrated[0][0] == "book"
    assert runtime._cognitive_library_catalog() == []


def test_completed_read_proposal_requires_explicit_reread_reason(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(runtime_module, "library_catalog", lambda limit=20, **kwargs: [{
        "id": "book",
        "title": "Book",
        "fully_read": True,
    }])

    ordinary = ActionCandidate(
        kind="library",
        description="Read remaining sections",
        payload={"action": "read", "item_id": "book"},
        novelty=0.9,
        information_gain=0.9,
    )
    assert runtime._filter_library_candidates([ordinary]) == []

    deliberate = ActionCandidate(
        kind="library",
        description="Reread a passage for a new comparison",
        payload={
            "action": "read",
            "item_id": "book",
            "reread": True,
            "reason": "Compare its definition against a newly formed hypothesis.",
        },
        novelty=0.9,
        information_gain=0.9,
    )
    kept = runtime._filter_library_candidates([deliberate])
    assert kept == [deliberate]
    assert deliberate.novelty <= 0.18
    assert deliberate.information_gain <= 0.28


def test_library_inventory_keeps_late_wishes_and_collapses_legacy_duplicates(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    entries = [
        {"id": f"book-{i}", "title": f"Book {i}", "kind": "book",
         "fully_read": False, "content_available": True}
        for i in range(25)
    ] + [
        {"id": "wish-1", "title": "Tear physiology", "kind": "reading_request",
         "status": "wanted", "fully_read": False, "content_available": False},
        {"id": "wish-2", "title": "TEAR--PHYSIOLOGY", "kind": "reading_request",
         "status": "wanted", "fully_read": False, "content_available": False},
    ]
    monkeypatch.setattr(runtime_module, "library_catalog", lambda limit=20, **kwargs: entries[:limit])

    visible = runtime._cognitive_library_catalog()
    assert len(visible) == 26
    assert visible[-1]["id"] == "wish-1"
    assert any(entry["id"] == "book-24" for entry in visible)


def test_library_action_candidates_ignore_existing_wishes_and_unavailable_text(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    entries = [
        {"id": "wish", "title": "Tear physiology", "kind": "reading_request",
         "status": "wanted", "content_available": False},
        {"id": "book", "title": "Readable book", "kind": "book",
         "status": "available", "content_available": True},
    ]
    monkeypatch.setattr(runtime_module, "library_catalog", lambda limit=20, **kwargs: entries[:limit])
    repeated = ActionCandidate(kind="library", description="Same wish",
                               payload={"action": "request", "title": "TEAR PHYSIOLOGY"})
    new = ActionCandidate(kind="library", description="Different wish",
                          payload={"action": "request", "title": "Some new topic"})
    unavailable = ActionCandidate(kind="library", description="Read absent text",
                                  payload={"action": "read", "item_id": "wish"})
    readable = ActionCandidate(kind="library", description="Read available text",
                               payload={"action": "read", "item_id": "book"})
    assert runtime._filter_library_candidates([repeated, new, unavailable, readable]) == [
        new, readable
    ]


def test_recall_keeps_one_episode_per_unfulfilled_wish(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.external_issue_states = {}
    requests = [
        {"id": f"duplicate-{i}", "kind": "internal_action_outcome",
         "text": json.dumps({"action": "request", "item": {"title": "Tear physiology"}}),
         "recall_score": 0.95 - i * 0.03, "concepts": ["tears"]}
        for i in range(3)
    ]
    requests.append({"id": "different", "kind": "library_reading",
                     "text": "An independent book passage", "concepts": ["humanity"],
                     "recall_score": 0.6})
    recalled = runtime._contextualize_recall(requests, limit=4)
    assert [item["id"] for item in recalled] == ["duplicate-0", "different"]


def test_unavailable_book_returns_failed_action_not_success(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(runtime_module, "apply_library_action", lambda spec, actor: {
        "action": "read", "content_available": False, "excerpt": "",
        "item": {"id": spec["item_id"]}
    })
    candidate = ActionCandidate(kind="library", description="Read absent book",
                                payload={"action": "read", "item_id": "wish"})
    success, result, provider = runtime._act(candidate, generation=1, development_allowed=True)
    assert not success
    assert json.loads(result)["content_available"] is False
