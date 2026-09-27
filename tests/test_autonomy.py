from ubique.autonomy import autonomous_task


def test_without_remote_provider_uses_cheap_status_cycle():
    task = autonomous_task(6, remote_reasoning_available=False)
    assert task.source == "autonomous"
    assert task.body.startswith("/status")


def test_healthy_remote_cycle_reflects_instead_of_idling():
    task = autonomous_task(24, remote_reasoning_available=True)
    assert task.body.startswith("/reflect")
    assert "Layer 2" in task.body


def test_attention_can_request_evidence_driven_evolution():
    task = autonomous_task(
        25,
        remote_reasoning_available=True,
        cognition={"attention": {"next_command": "evolve", "hypothesis": "router can be simplified safely"}},
    )
    assert task.body.startswith("/evolve")
    assert "hypothesis" in task.body.lower()


def test_attention_can_request_question_driven_fzg():
    task = autonomous_task(
        26,
        remote_reasoning_available=True,
        cognition={"attention": {"next_command": "fzg", "hypothesis": "causal evidence is weak"}},
    )
    assert task.body.startswith("/fzg")


def test_blocked_preflight_prioritizes_layer_one():
    task = autonomous_task(
        27,
        remote_reasoning_available=True,
        preflight={"development_allowed": False},
    )
    assert task.body.startswith("/think")


def test_healthy_reflection_hides_layer_one_details():
    task = autonomous_task(
        30,
        remote_reasoning_available=True,
        homeostasis={"needs": [{"name": "cycle_reliability", "level": "stable", "value": 0.95}]},
        telemetry={"provider_count": 2},
        environment={"branch": "main", "configured_remote_providers": ["gemini", "groq"]},
        preflight={"development_allowed": True},
        cognition={"attention": {}, "projects": {"projects": []}, "curiosity": {"frontiers": []}},
    )
    assert task.body.startswith("/reflect")
    assert "Layer 1 status: healthy" in task.body
    assert "cycle_reliability" not in task.body
    assert "configured_remote_providers" not in task.body
