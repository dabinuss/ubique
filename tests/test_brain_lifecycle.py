from ubique.brain.episodic import EpisodeStore
from ubique.brain.network import AssociativeNetwork
from ubique.brain.runtime import NeurocognitiveRuntime


class _ClosedIssueGitHub:
    def issue_state(self, issue_number: int) -> dict:
        return {
            "number": issue_number,
            "title": "Heartbeat smoke test",
            "state": "closed",
            "labels": [],
            "updated_at": "2026-10-02T00:00:00+00:00",
            "closed_at": "2026-10-02T00:00:00+00:00",
        }


def test_closed_issue_becomes_terminal_percept_and_inhibits_old_assembly(tmp_path):
    runtime = NeurocognitiveRuntime.__new__(NeurocognitiveRuntime)
    runtime.episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    runtime.network = AssociativeNetwork(tmp_path / "cortex.json")
    runtime.github = _ClosedIssueGitHub()
    runtime.external_issue_states = {}

    runtime.episodes.append(
        kind="github_issue",
        text="Heartbeat smoke test",
        source="github:test",
        concepts=["heartbeat", "smoke", "test"],
        epistemic_status="observed_external_input",
        payload={"issue_number": 11},
        generation=1,
    )
    heartbeat = runtime.network.ensure_node("heartbeat")
    stability = runtime.network.ensure_node("runtime stability")
    runtime.network.connect(heartbeat.id, stability.id, weight=0.8)
    runtime.network.activate_ids([heartbeat.id, stability.id], amount=0.9)
    before_neighbor = runtime.network.nodes[stability.id].activation

    percepts = runtime._observe_issue_lifecycle([], generation=2)

    assert len(percepts) == 1
    assert percepts[0]["kind"] == "external_issue_lifecycle"
    assert percepts[0]["payload"]["lifecycle"] == "closed"
    assert runtime.external_issue_states["11"]["state"] == "closed"
    assert "heartbeat" in runtime.external_issue_states["11"]["concepts"]
    assert runtime.network.nodes[heartbeat.id].activation < 0.2
    assert runtime.network.nodes[stability.id].activation < before_neighbor

    runtime.network.activate_ids([heartbeat.id, stability.id], amount=0.9)
    reactivated = runtime.network.nodes[stability.id].activation
    touched = runtime._apply_terminal_inhibition()
    assert touched >= 2
    assert runtime.network.nodes[heartbeat.id].activation < 0.3
    assert runtime.network.nodes[stability.id].activation < reactivated
