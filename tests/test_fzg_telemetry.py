import ubique.fzg_telemetry as ft


def test_fzg_telemetry_is_profile_not_global_score(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "write_json", lambda name, value: None)
    monkeypatch.setattr(ft, "read_json", lambda name, default: {})
    monkeypatch.setattr(ft, "recent_episodes", lambda limit=100: [])
    out = ft.measure_fzg_telemetry()
    assert set(out["I_C"]) == {"Z", "K", "R", "L"}
    assert "global_score" not in out
    assert out["G_A"]["ablation_required_for_causal_claim"] is True


def test_redundancy_proxy_targets_one_provider_loss(monkeypatch):
    monkeypatch.setattr(
        ft,
        "read_json",
        lambda name, default: {
            "gemini": {"disabled_until": None},
            "groq": {"disabled_until": None},
            "fallback": {"disabled_until": None},
        },
    )
    out = ft.controlled_ablation_proxy()
    assert out["healthy_remote_providers"] == 2
    assert out["Q_M_S"] == 1.0
    assert out["Q_M_S_minus"] == 0.0
    assert out["G_A_proxy"] == 1.0
    assert "loss of one" in out["A"]


def test_single_provider_has_no_redundancy_advantage(monkeypatch):
    monkeypatch.setattr(
        ft,
        "read_json",
        lambda name, default: {
            "gemini": {"disabled_until": None},
            "fallback": {"disabled_until": None},
        },
    )
    out = ft.controlled_ablation_proxy()
    assert out["healthy_remote_providers"] == 1
    assert out["G_A_proxy"] == 0.0
