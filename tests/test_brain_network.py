from ubique.brain.network import AssociativeNetwork


def test_spreading_activation_and_hebbian_strengthening(tmp_path):
    network = AssociativeNetwork(tmp_path / "cortex.json")
    left = network.ensure_node("memory", kind="concept")
    right = network.ensure_node("identity", kind="concept")
    network.connect(left.id, right.id, weight=0.6)
    network.activate_ids([left.id], amount=0.95)

    network.spread(steps=1, gain=0.8, decay=0.9)
    assert network.nodes[right.id].activation > 0.0

    key = network.edge_id(left.id, right.id, "coactivated")
    network.hebbian_update([left.id, right.id], learning_rate=0.2)
    first = network.edges[key].weight
    network.hebbian_update([left.id, right.id], learning_rate=0.2)
    assert network.edges[key].weight > first

    before = network.nodes[left.id].activation
    network.decay(activation_factor=0.5)
    assert network.nodes[left.id].activation < before


def test_homeostatic_normalization_prevents_mass_saturation(tmp_path):
    network = AssociativeNetwork(tmp_path / "cortex.json")
    ids = network.activate_labels([f"node-{i}" for i in range(20)], amount=1.0)
    assert ids
    network.homeostatic_normalize(target_mean=0.25, ceiling=0.88)
    activations = [network.nodes[node_id].activation for node_id in ids]
    assert max(activations) <= 0.88
    assert sum(activations) / len(activations) <= 0.251


def test_inhibition_reduces_resolved_assembly_and_neighbors(tmp_path):
    network = AssociativeNetwork(tmp_path / "cortex.json")
    left = network.ensure_node("heartbeat")
    right = network.ensure_node("runtime stability")
    network.connect(left.id, right.id, weight=0.8)
    network.activate_ids([left.id, right.id], amount=0.9)
    before = network.nodes[right.id].activation
    touched = network.inhibit_labels(["heartbeat"], factor=0.1, neighbor_factor=0.2)
    assert touched >= 2
    assert network.nodes[left.id].activation < 0.2
    assert network.nodes[right.id].activation < before


def test_pruning_caps_schema_and_edge_growth(tmp_path):
    network = AssociativeNetwork(tmp_path / "cortex.json")
    anchors = [network.ensure_node(f"anchor-{i}") for i in range(8)]
    for i in range(80):
        schema = network.ensure_node(f"schema-{i}", kind="schema")
        for anchor in anchors:
            network.connect(schema.id, anchor.id, weight=0.2, bidirectional=False)
    stats = network.prune(max_schema_nodes=12, max_edges=60)
    assert sum(node.kind == "schema" for node in network.nodes.values()) <= 12
    assert len(network.edges) <= 60
    assert stats["removed_nodes"] > 0
