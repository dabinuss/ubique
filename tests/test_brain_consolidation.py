from ubique.brain.consolidation import Consolidator
from ubique.brain.episodic import EpisodeStore
from ubique.brain.network import AssociativeNetwork


def test_nrem_replay_builds_repeated_schema(tmp_path):
    episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    network = AssociativeNetwork(tmp_path / "cortex.json")
    for text in ("memory replay changed learning", "memory replay returned during rest"):
        episodes.append(
            kind="experience",
            text=text,
            source="test",
            concepts=["memory", "replay"],
            salience=0.9,
            surprise=0.6,
        )

    result = Consolidator(
        episodes,
        network,
        substrate=None,
        simulations_path=tmp_path / "simulations.jsonl",
    ).nrem()

    assert result["mode"] == "nrem"
    assert result["schema_nodes_reinforced"] >= 1
    assert any(node.kind == "schema" for node in network.nodes.values())


def test_rem_is_explicitly_imagined(tmp_path):
    episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    network = AssociativeNetwork(tmp_path / "cortex.json")
    network.activate_labels(["memory", "future"], amount=0.8)
    path = tmp_path / "simulations.jsonl"

    result = Consolidator(
        episodes,
        network,
        substrate=None,
        simulations_path=path,
    ).rem()

    assert result["mode"] == "rem"
    assert '"epistemic_status": "imagined"' in path.read_text(encoding="utf-8")
