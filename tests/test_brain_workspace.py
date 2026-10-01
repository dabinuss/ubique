from ubique.brain.workspace import GlobalWorkspace


def test_workspace_is_capacity_limited_and_competitive():
    workspace = GlobalWorkspace(capacity=3)
    items = workspace.compete([
        {"id": f"n{i}", "label": f"node {i}", "activation": i / 10, "salience": i / 10}
        for i in range(1, 8)
    ])
    assert len(items) == 3
    assert items[0].label == "node 7"
    assert items[-1].label == "node 5"
