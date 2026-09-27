import ubique.fzg_telemetry as ft


def test_fzg_telemetry_is_profile_not_global_score(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "write_json", lambda name, value: None)
    monkeypatch.setattr(ft, "read_json", lambda name, default: {})
    monkeypatch.setattr(ft, "recent_episodes", lambda limit=100: [])
    out = ft.measure_fzg_telemetry()
    assert set(out["I_C"]) == {"Z", "K", "R", "L"}
    assert "global_score" not in out
    assert out["G_A"]["ablation_required_for_causal_claim"] is True
