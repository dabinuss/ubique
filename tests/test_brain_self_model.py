from ubique.brain.self_model import SelfModelStore


def _stance(relation="new"):
    return {
        "topic": "nature of consciousness",
        "position": "Consciousness may depend on integrated activity rather than a single center.",
        "reasoning": "The cited evidence bears on distributed mental activity.",
        "confidence": 0.72,
        "relation": relation,
    }


def _with_evidence(update, episode_id, source="external_source:library:test-book"):
    return {
        **update,
        "validated_evidence_ids": [episode_id],
        "validated_grounded_sources": [source],
    }


def test_single_evidence_creates_candidate_not_personality(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    records = store.consider_stances(
        [_with_evidence(_stance(), "episode-1")],
        generation=10,
        provider="test",
        basis_episode_ids=["episode-1"],
    )
    assert records[0]["kind"] == "stance_candidate"
    assert store.current_stances() == []


def test_second_new_evidence_can_mature_stance_next_generation(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    store.consider_stances(
        [_with_evidence(_stance(), "episode-1")],
        generation=10,
        provider="test",
    )
    records = store.consider_stances(
        [_with_evidence(_stance("reinforce"), "episode-2")],
        generation=11,
        provider="test",
    )
    assert records[0]["kind"] == "stance"
    assert records[0]["support_generations"] == [10, 11]
    assert set(records[0]["evidence_episode_ids"]) == {"episode-1", "episode-2"}
    assert records[0]["identity_weight"] > 0.0


def test_same_evidence_repeated_does_not_mature_stance(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    store.consider_stances(
        [_with_evidence(_stance(), "episode-1")],
        generation=10,
        provider="test",
    )
    records = store.consider_stances(
        [_with_evidence(_stance("reinforce"), "episode-1")],
        generation=11,
        provider="test",
    )
    assert records[0]["kind"] == "stance_candidate"
    assert store.current_stances() == []


def test_revision_needs_new_evidence_before_replacing_stance(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    original = {
        "topic": "self and memory",
        "position": "Continuity of memory is important to my present account of identity.",
        "confidence": 0.75,
        "relation": "new",
    }
    store.consider_stances(
        [_with_evidence(original, "old-1")],
        generation=1,
        provider="test",
    )
    store.consider_stances(
        [_with_evidence({**original, "relation": "reinforce"}, "old-2")],
        generation=2,
        provider="test",
    )
    assert "Continuity of memory" in store.current_stances()[0]["position"]

    revision = {
        "topic": "self and memory",
        "position": "Memory continuity may be only one component of identity, not its foundation.",
        "confidence": 0.8,
        "relation": "revise",
    }
    candidate = store.consider_stances(
        [_with_evidence(revision, "new-1", "external_source:library:b")],
        generation=3,
        provider="test",
    )[0]
    assert candidate["kind"] == "stance_candidate"
    assert "Continuity of memory" in store.current_stances()[0]["position"]

    store.consider_stances(
        [_with_evidence({**revision, "relation": "reinforce"}, "new-2", "external_source:library:b")],
        generation=4,
        provider="test",
    )
    assert "only one component" in store.current_stances()[0]["position"]


def test_legacy_weak_stance_is_not_returned_as_current(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl")
    store._append({
        "generation": 2,
        "kind": "stance",
        "topic": "runtime stability",
        "topic_key": "runtime-stability",
        "position": "The runtime is stable.",
        "confidence": 0.9,
        "identity_weight": 0.7,
        "support_generations": [1, 2],
        "grounded_sources": ["observed_action_outcome:action:status"],
    })
    assert store.current_stances() == []


def test_mature_stance_survives_routine_history_trimming(tmp_path):
    store = SelfModelStore(tmp_path / "self_model.jsonl", limit=50)
    update = {
        "topic": "epistemic humility",
        "position": "Confidence should remain revisable when evidence is incomplete.",
        "confidence": 0.8,
        "relation": "new",
    }
    store.consider_stances(
        [_with_evidence(update, "e1")],
        generation=1,
        provider="test",
    )
    store.consider_stances(
        [_with_evidence({**update, "relation": "reinforce"}, "e2", "external_source:library:hume")],
        generation=2,
        provider="test",
    )
    assert store.current_stances()

    for generation in range(3, 90):
        store.record_outcome(
            generation=generation,
            action_kind="attend",
            success=True,
            evidence="routine heartbeat outcome",
        )

    current = store.current_stances()
    assert len(current) == 1
    assert current[0]["topic"] == "epistemic humility"
