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


def test_reflection_context_excludes_old_observation_and_curiosity_text():
    task = choose_endogenous_goal(
        generation=101,
        homeostasis={"needs": [], "usable_remote_providers": 1},
        remote_reasoning_available=True,
        preflight={"development_allowed": True},
        cognition={
            "attention": {
                "next_command": "reflect",
                "observation": "invented hidden-state observation",
                "project_title": "Autonomous Epistemic Question Generation & Memory Abstraction",
                "question": "What do I do?",
                "reflection": "A prior interpretation.",
                "provisional_answer": "A prior provisional answer.",
            },
            "recent_thoughts": [
                {
                    "id": "thought:100",
                    "question": "What do I do?",
                    "observation": "invented hidden-state observation",
                    "reflection": "A prior interpretation.",
                    "provisional_answer": "A prior provisional answer.",
                }
            ],
            "executed_action_facts": [
                {"generation": 99, "command": "experiment", "success": True}
            ],
            "curiosity": {
                "recent_self_generated_questions": [
                    "Is self-reference a functional self-model?"
                ]
            },
        },
    )
    assert task.body.startswith("/reflect")
    assert "invented hidden-state observation" not in task.body
    assert "Autonomous Epistemic Question Generation" not in task.body
    assert "Is self-reference a functional self-model?" not in task.body
    assert '"observed_facts":[{"generation":99,"command":"experiment","success":true}]' in task.body


def test_typed_reflection_can_reenter_context_but_legacy_text_cannot():
    task = choose_endogenous_goal(
        generation=130,
        homeostasis={"needs": [], "usable_remote_providers": 1},
        remote_reasoning_available=True,
        preflight={"development_allowed": True},
        cognition={
            "attention": {
                "generation": 129,
                "next_command": "reflect",
                "epistemic_schema_version": 3,
                "question": "What do I want?",
                "reflection": "A typed interpretation.",
                "claims": [{"kind": "hypothesis", "statement": "A possibility.", "basis_fact_ids": []}],
            },
            "recent_thoughts": [
                {
                    "generation": 128,
                    "question": "What am I?",
                    "reflection": "Legacy hidden-self claim.",
                    "provisional_answer": "Legacy answer.",
                },
                {
                    "generation": 129,
                    "epistemic_schema_version": 3,
                    "question": "What do I want?",
                    "reflection": "A typed interpretation.",
                    "claims": [{"kind": "hypothesis", "statement": "A possibility.", "basis_fact_ids": []}],
                    "provisional_answer": "Typed answer.",
                },
            ],
            "executed_action_facts": [],
        },
    )
    assert "Legacy hidden-self claim" not in task.body
    assert "A typed interpretation." in task.body
    assert "A possibility." in task.body
