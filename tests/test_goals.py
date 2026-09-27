from ubique.goals import choose_endogenous_goal


def test_unavailable_provider_probe_is_replanned_as_reflection():
    task = choose_endogenous_goal(
        generation=64,
        homeostasis={"needs": [], "usable_remote_providers": 1},
        remote_reasoning_available=True,
        preflight={"development_allowed": True},
        cognition={
            "attention": {
                "next_command": "experiment",
                "hypothesis": "test",
                "experiment_type": "provider_probe",
                "experiment_target": "gemini",
            },
            "provider_eligibility": {
                "gemini": {"eligible": False, "remaining_calls": 0},
                "groq": {"eligible": True, "remaining_calls": 900},
            },
        },
    )
    assert task.body.startswith("/reflect")
    assert "Do not retry that provider" in task.body
    assert "non-provider experiment" in task.body


def test_eligible_provider_probe_remains_experiment():
    task = choose_endogenous_goal(
        generation=64,
        homeostasis={"needs": [], "usable_remote_providers": 2},
        remote_reasoning_available=True,
        preflight={"development_allowed": True},
        cognition={
            "attention": {
                "next_command": "experiment",
                "hypothesis": "test provider path",
                "experiment_type": "provider_probe",
                "experiment_target": "groq",
            },
            "provider_eligibility": {
                "groq": {"eligible": True, "remaining_calls": 900},
            },
        },
    )
    assert task.body.startswith("/experiment")


def test_resolution_is_scheduled_before_more_experimentation():
    task = choose_endogenous_goal(
        generation=95,
        homeostasis={"needs": [], "usable_remote_providers": 1},
        remote_reasoning_available=True,
        preflight={"development_allowed": True},
        cognition={
            "attention": {
                "next_command": "resolve",
                "question": "Did the experiment answer Q?",
                "hypothesis": "H",
                "last_evidence": "observed result",
            },
        },
    )
    assert task.body.startswith("/resolve")
    assert "Resolve the current question" in task.body
