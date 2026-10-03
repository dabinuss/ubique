from ubique.brain.episodic import EpisodeStore


def test_associative_recall_can_beat_recency(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.jsonl", limit=200)
    old = store.append(
        kind="experience",
        text="A strange memory consolidation event connected hippocampus and replay.",
        source="test",
        concepts=["memory", "consolidation", "replay"],
        salience=0.7,
    )
    for index in range(8):
        store.append(
            kind="experience",
            text=f"Unrelated recent weather record number {index}.",
            source="test",
            concepts=["weather"],
            salience=0.2,
        )

    recalled = store.recall(
        "How does memory replay support consolidation?",
        concepts=["memory", "replay"],
        limit=2,
    )
    assert recalled
    assert recalled[0]["id"] == old["id"]


def test_imagined_memory_can_be_excluded(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.jsonl", limit=200)
    store.append(
        kind="simulation",
        text="imaginary bridge between two concepts",
        source="test",
        concepts=["bridge"],
        epistemic_status="imagined",
    )
    assert store.recall("bridge", concepts=["bridge"], include_imagined=False) == []


def test_imagined_memory_is_weaker_than_grounded_memory_with_same_topic(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.jsonl", limit=200)
    grounded = store.append(
        kind="experience",
        text="A grounded observation about memory, identity, and attention.",
        source="test",
        concepts=["memory", "identity", "attention"],
        epistemic_status="observed_action_outcome",
        salience=0.55,
        surprise=0.3,
    )
    store.append(
        kind="simulation",
        text="An imagined counterfactual about memory, identity, and attention.",
        source="test",
        concepts=["memory", "identity", "attention"],
        epistemic_status="imagined",
        salience=1.0,
        surprise=1.0,
    )

    recalled = store.recall(
        "memory identity attention",
        concepts=["memory", "identity", "attention"],
        limit=2,
        include_imagined=True,
    )
    assert recalled[0]["id"] == grounded["id"]
