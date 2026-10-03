from ubique.brain.runtime import NeurocognitiveRuntime


def test_library_and_observed_action_can_ground_stances():
    records = [
        {
            "epistemic_status": "external_source",
            "source": "library:darwin-expression-emotions",
        },
        {
            "epistemic_status": "observed_action_outcome",
            "source": "action:experiment",
        },
    ]
    assert NeurocognitiveRuntime._stance_grounding_sources(records) == [
        "external_source:library:darwin-expression-emotions",
        "observed_action_outcome:action:experiment",
    ]


def test_issue_text_and_lifecycle_cannot_ground_personality():
    records = [
        {
            "epistemic_status": "observed_external_input",
            "source": "github:user",
        },
        {
            "epistemic_status": "observed_external_state",
            "source": "github:lifecycle",
        },
    ]
    assert NeurocognitiveRuntime._stance_grounding_sources(records) == []
