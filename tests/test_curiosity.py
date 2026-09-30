import ubique.curiosity as curiosity


def test_curiosity_prioritizes_unmentioned_frontiers(monkeypatch):
    monkeypatch.setattr(curiosity, "recent_memory_records", lambda name, limit=20: [])
    writes = {}
    monkeypatch.setattr(curiosity, "write_json", lambda name, value: writes.__setitem__(name, value))
    out = curiosity.build_curiosity_snapshot(7)
    assert out["generation"] == 7
    assert out["frontiers"]
    assert all(item["novelty_priority"] == 1.0 for item in out["frontiers"])
    assert "curiosity.json" in writes


def test_curiosity_does_not_replay_old_generated_questions(monkeypatch):
    monkeypatch.setattr(
        curiosity,
        "recent_memory_records",
        lambda name, limit=20: [
            {
                "question": "Is self-reference a functional self-model?",
                "provisional_answer": "old answer",
            }
        ],
    )
    monkeypatch.setattr(curiosity, "write_json", lambda name, value: None)
    out = curiosity.build_curiosity_snapshot(8)
    assert out["recent_self_generated_questions"] == []
    assert "Old model-generated question text is intentionally not replayed" in out["rule"]
