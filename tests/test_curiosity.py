import ubique.curiosity as curiosity


def test_curiosity_prioritizes_unmentioned_frontiers(tmp_path, monkeypatch):
    monkeypatch.setattr(curiosity, "ROOT", tmp_path)
    (tmp_path / "src" / "ubique").mkdir(parents=True)
    (tmp_path / "src" / "ubique" / "planner.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(curiosity, "recent_memory_records", lambda name, limit=12: [])
    writes = {}
    monkeypatch.setattr(curiosity, "write_json", lambda name, value: writes.__setitem__(name, value))
    out = curiosity.build_curiosity_snapshot(7)
    assert out["generation"] == 7
    assert out["frontiers"]
    assert all(item["novelty_priority"] == 1.0 for item in out["frontiers"])
    assert "curiosity.json" in writes


def test_curiosity_is_explicitly_non_homeostatic(monkeypatch, tmp_path):
    monkeypatch.setattr(curiosity, "ROOT", tmp_path)
    (tmp_path / "src" / "ubique").mkdir(parents=True)
    monkeypatch.setattr(curiosity, "recent_memory_records", lambda name, limit=12: [])
    monkeypatch.setattr(curiosity, "write_json", lambda name, value: None)
    out = curiosity.build_curiosity_snapshot(8)
    assert "operational continuity is out of scope" in out["rule"]
