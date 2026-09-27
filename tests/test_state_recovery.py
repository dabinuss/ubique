import ubique.state as state


def test_corrupt_state_falls_back_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert state.read_json("broken.json", {"safe": True}) == {"safe": True}
