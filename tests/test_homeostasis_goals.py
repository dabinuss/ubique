import ubique.homeostasis as homeostasis
from ubique.goals import choose_endogenous_goal
from ubique.fzg_telemetry import controlled_ablation_proxy


def test_critical_need_becomes_diagnosis_goal():
    task = choose_endogenous_goal(
        7,
        {"needs": [{"name": "cycle_reliability", "level": "critical", "value": 0.2}]},
        True,
    )
    assert task.body.startswith("/think")
    assert "cycle_reliability" in task.body


def test_stable_generation_24_can_evolve():
    task = choose_endogenous_goal(
        24,
        {"needs": [{"name": "cycle_reliability", "level": "stable", "value": 1.0}]},
        True,
    )
    assert task.body.startswith("/evolve")


def test_watch_need_requests_fzg_measurement():
    task = choose_endogenous_goal(
        7,
        {"needs": [{"name": "reasoning_redundancy", "level": "watch", "value": 1.0}]},
        True,
    )
    assert task.body.startswith("/fzg")


def test_no_remote_reasoning_remains_deterministic():
    task = choose_endogenous_goal(
        24,
        {"needs": [{"name": "cycle_reliability", "level": "critical", "value": 0.2}]},
        False,
    )
    assert task.body.startswith("/status")


def test_ablation_proxy_declares_identification_limits():
    out = controlled_ablation_proxy()
    assert "M_S" in out and "M_S_minus" in out
    assert "proxy" in out["identification"]


def test_memory_pressure_uses_full_retained_count(monkeypatch):
    monkeypatch.setattr(homeostasis, "recent_episodes", lambda limit=24: [{"success": True}])
    monkeypatch.setattr(homeostasis, "episode_count", lambda: 400)
    monkeypatch.setattr(homeostasis, "read_json", lambda name, default: {})
    monkeypatch.setattr(homeostasis, "write_json", lambda name, value: None)
    out = homeostasis.assess_homeostasis(memory_limit=500, configured_remote=["gemini", "groq"])
    memory_need = next(n for n in out["needs"] if n["name"] == "memory_pressure")
    assert memory_need["level"] == "watch"
    assert memory_need["value"] == 400.0
    assert out["retained_episode_count"] == 400


def test_memory_pressure_becomes_critical_at_limit(monkeypatch):
    monkeypatch.setattr(homeostasis, "recent_episodes", lambda limit=24: [{"success": True}])
    monkeypatch.setattr(homeostasis, "episode_count", lambda: 500)
    monkeypatch.setattr(homeostasis, "read_json", lambda name, default: {})
    monkeypatch.setattr(homeostasis, "write_json", lambda name, value: None)
    out = homeostasis.assess_homeostasis(memory_limit=500, configured_remote=["gemini", "groq"])
    memory_need = next(n for n in out["needs"] if n["name"] == "memory_pressure")
    assert memory_need["level"] == "critical"
