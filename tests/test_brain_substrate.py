from ubique.brain.substrate import cognitive_prompt


def test_cognitive_prompt_marks_terminal_state_as_current_observation():
    prompt = cognitive_prompt(
        workspace="- [concept/model_proposal] heartbeat persists",
        recalled_episodes=[{
            "id": "old",
            "kind": "github_issue",
            "epistemic_status": "observed_external_input",
            "text": "heartbeat smoke test is active",
            "concepts": ["heartbeat"],
        }],
        modulators={"energy": 0.8},
        self_model=[{
            "epistemic_status": "model_interpretation",
            "statement": "the test may still be in progress",
            "confidence": 0.5,
        }],
        library_catalog=[],
        external_states={
            "11": {
                "state": "closed",
                "title": "Heartbeat smoke test",
                "closed_at": "2026-10-02T00:00:00+00:00",
            }
        },
    )
    assert '"state": "closed"' in prompt
    assert "supersede" in prompt
    assert "Do not continue a terminal task" in prompt
