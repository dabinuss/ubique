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
