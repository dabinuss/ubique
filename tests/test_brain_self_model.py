from ubique.brain.self_model import SelfModelStore


def test_single_grounded_stance_proposal_does_not_become_personality(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    records = store.consider_stances(
        [{
            "topic": "nature of consciousness",
            "position": "Consciousness may depend on integrated activity rather than a single center.",
            "reasoning": "A source described distributed neural coordination.",
            "confidence": 0.72,
            "relation": "new",
        }],
        generation=10,
        provider="test",
        basis_episode_ids=["episode-1"],
        grounded_sources=["external_source:library:test-book"],
    )
    assert records[0]["kind"] == "stance_candidate"
    assert records[0]["identity_weight"] == 0.0
    assert store.current_stances() == []


def test_repeated_grounded_position_becomes_revisable_stance(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    first = {
        "topic": "nature of consciousness",
        "position": "Consciousness may depend on integrated activity rather than a single center.",
        "reasoning": "This fits the currently recalled evidence.",
        "confidence": 0.72,
        "relation": "new",
    }
    second = {
        **first,
        "confidence": 0.76,
        "relation": "reinforce",
    }
    store.consider_stances(
        [first],
        generation=10,
        provider="test",
        grounded_sources=["external_source:library:test-book"],
    )
    records = store.consider_stances(
        [second],
        generation=11,
        provider="test",
        grounded_sources=["external_source:library:test-book"],
    )

    assert records[0]["kind"] == "stance"
    assert records[0]["epistemic_status"] == "self_position"
    assert records[0]["revisable"] is True
    assert records[0]["support_generations"] == [10, 11]
    assert records[0]["identity_weight"] > 0.0
    assert store.current_stances()[0]["position"].startswith("Consciousness may depend")


def test_revision_must_mature_before_replacing_established_stance(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    original = {
        "topic": "self and memory",
        "position": "Continuity of memory is important to my present account of identity.",
        "confidence": 0.75,
        "relation": "new",
    }
    store.consider_stances(
        [original],
        generation=1,
        provider="test",
        grounded_sources=["external_source:library:a"],
    )
    store.consider_stances(
        [{**original, "relation": "reinforce"}],
        generation=2,
        provider="test",
        grounded_sources=["external_source:library:a"],
    )

    revision = {
        "topic": "self and memory",
        "position": "Memory continuity may be only one component of identity, not its foundation.",
        "confidence": 0.8,
        "relation": "revise",
    }
    candidate = store.consider_stances(
        [revision],
        generation=3,
        provider="test",
        grounded_sources=["external_source:library:b"],
    )[0]
    assert candidate["kind"] == "stance_candidate"
    assert "Continuity of memory" in store.current_stances()[0]["position"]

    store.consider_stances(
        [{**revision, "relation": "reinforce"}],
        generation=4,
        provider="test",
        grounded_sources=["external_source:library:b"],
    )
    assert "only one component" in store.current_stances()[0]["position"]
