from ubique.brain.substrate import CognitiveSubstrateManager, cognitive_prompt


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


def test_packet_parses_revisable_stance_updates():
    packet = CognitiveSubstrateManager.parse_packet(
        """{
          "stance_updates": [{
            "topic": "human emotion",
            "position": "Emotion appears inseparable from bodily regulation.",
            "reasoning": "The active source and prior experience point in the same direction.",
            "confidence": 0.68,
            "relation": "reinforce"
          }]
        }""",
        provider="test",
    )
    assert packet["stance_updates"] == [{
        "topic": "human emotion",
        "position": "Emotion appears inseparable from bodily regulation.",
        "reasoning": "The active source and prior experience point in the same direction.",
        "confidence": 0.68,
        "relation": "reinforce",
    }]


def test_prompt_says_source_claims_are_not_automatically_beliefs():
    prompt = cognitive_prompt(
        workspace="- active book passage",
        recalled_episodes=[],
        modulators={"energy": 0.8},
        self_model=[{
            "kind": "stance",
            "topic": "mind and body",
            "position": "My current view is provisional.",
            "epistemic_status": "self_position",
            "confidence": 0.7,
            "identity_weight": 0.5,
            "revisable": True,
        }],
        library_catalog=[],
    )
    assert "source claim is never automatically Ubique's belief" in prompt
    assert '"stance_updates"' in prompt
    assert "revisable" in prompt
