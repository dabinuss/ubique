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
