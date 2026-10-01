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
