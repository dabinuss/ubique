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


def test_terminal_context_downweights_stale_model_recall(tmp_path):
    runtime = NeurocognitiveRuntime.__new__(NeurocognitiveRuntime)
    runtime.episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    runtime.network = AssociativeNetwork(tmp_path / "cortex.json")
    runtime.external_issue_states = {
        "11": {
            "state": "closed",
            "concepts": ["heartbeat", "smoke", "test"],
        }
    }

    heartbeat = runtime.network.ensure_node("heartbeat")
    stale = runtime.network.ensure_node("heartbeat persists")
    runtime.network.connect(heartbeat.id, stale.id, weight=0.9)

    recalled = [
        {
            "id": "stale",
            "kind": "cognitive_packet",
            "epistemic_status": "model_proposal",
            "concepts": ["heartbeat persists"],
            "recall_score": 0.9,
            "text": "old model packet",
        },
        {
            "id": "fresh",
            "kind": "library_reading",
            "epistemic_status": "external_source",
            "concepts": ["memory", "learning"],
            "recall_score": 0.5,
            "text": "unrelated external evidence",
        },
    ]

    result = runtime._contextualize_recall(recalled, limit=2)

    assert result[0]["id"] == "fresh"
    stale_result = next(item for item in result if item["id"] == "stale")
    assert stale_result["contextual_status"] == "superseded_terminal_context"
    assert stale_result["recall_score"] < 0.5


def test_terminal_context_can_occupy_only_one_recall_slot(tmp_path):
    runtime = NeurocognitiveRuntime.__new__(NeurocognitiveRuntime)
    runtime.episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    runtime.network = AssociativeNetwork(tmp_path / "cortex.json")
    runtime.external_issue_states = {
        "11": {"state": "closed", "concepts": ["heartbeat"]}
    }
    heartbeat = runtime.network.ensure_node("heartbeat")
    for label in ("heartbeat persists", "runtime stability", "smoke test"):
        node = runtime.network.ensure_node(label)
        runtime.network.connect(heartbeat.id, node.id, weight=0.9)

    recalled = [
        {
            "id": f"stale-{index}",
            "kind": "cognitive_packet",
            "epistemic_status": "model_proposal",
            "concepts": [label],
            "recall_score": 0.95 - index * 0.01,
            "text": label,
        }
        for index, label in enumerate(("heartbeat persists", "runtime stability", "smoke test"))
    ] + [
        {
            "id": "other-1",
            "kind": "library_reading",
            "epistemic_status": "external_source",
            "concepts": ["learning"],
            "recall_score": 0.4,
            "text": "learning",
        },
        {
            "id": "other-2",
            "kind": "internal_action_outcome",
            "epistemic_status": "observed_action_outcome",
            "concepts": ["rest"],
            "recall_score": 0.35,
            "text": "rest",
        },
    ]

    result = runtime._contextualize_recall(recalled, limit=4)
    superseded = [
        item for item in result
        if item.get("contextual_status") == "superseded_terminal_context"
    ]
    assert len(superseded) <= 1
    assert {"other-1", "other-2"}.issubset({item["id"] for item in result})


def test_terminal_recall_matching_survives_missing_graph_edge(tmp_path):
    runtime = NeurocognitiveRuntime.__new__(NeurocognitiveRuntime)
    runtime.episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    runtime.network = AssociativeNetwork(tmp_path / "cortex.json")
    runtime.external_issue_states = {
        "11": {
            "state": "closed",
            "concepts": ["heartbeat", "persists", "smoke", "test"],
        }
    }

    # Deliberately do not create a graph edge from terminal seeds to the
    # model-authored compound concept. Pruning may legitimately remove it.
    recalled = [{
        "id": "stale",
        "kind": "cognitive_packet",
        "epistemic_status": "model_proposal",
        "concepts": ["heartbeat persists", "v2 runtime stability"],
        "recall_score": 0.9,
        "text": "old model packet",
    }]

    result = runtime._contextualize_recall(recalled, limit=1)

    assert result[0]["contextual_status"] == "superseded_terminal_context"
    assert result[0]["terminal_graph_overlap"] == 0
    assert result[0]["terminal_lexical_overlap"] >= 2
    assert result[0]["recall_score"] < 0.2


def test_terminal_inhibition_directly_suppresses_compound_model_nodes(tmp_path):
    runtime = NeurocognitiveRuntime.__new__(NeurocognitiveRuntime)
    runtime.episodes = EpisodeStore(tmp_path / "episodes.jsonl")
    runtime.network = AssociativeNetwork(tmp_path / "cortex.json")
    runtime.external_issue_states = {
        "11": {
            "state": "closed",
            "concepts": ["heartbeat", "persists", "smoke", "test"],
        }
    }

    stale = runtime.network.ensure_node(
        "heartbeat persists",
        kind="concept",
        epistemic_status="model_proposal",
    )
    unrelated = runtime.network.ensure_node(
        "library curiosity",
        kind="concept",
        epistemic_status="model_proposal",
    )
    runtime.network.activate_ids([stale.id, unrelated.id], amount=0.9)
    stale_before = stale.activation
    unrelated_before = unrelated.activation

    touched = runtime._apply_terminal_inhibition()

    assert touched >= 1
    assert stale.activation < stale_before * 0.3
    assert unrelated.activation == unrelated_before
