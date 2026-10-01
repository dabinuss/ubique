from ubique.brain.action_selection import ActionSelector
from ubique.brain.modulators import ModulatorState


def test_low_energy_system_can_choose_rest_or_consolidation():
    modulators = ModulatorState(
        novelty=0.05,
        surprise=0.05,
        uncertainty=0.15,
        salience=0.05,
        exploration=0.1,
        plasticity=0.2,
        energy=0.05,
        sleep_pressure=0.95,
    )
    selector = ActionSelector()
    candidates = selector.baseline_candidates(
        modulators,
        memory_count=120,
        workspace_activation=0.02,
    )
    selected, ranked = selector.select(candidates, modulators)
    assert selected.kind in {"rest", "consolidate"}
    assert ranked[0].score >= ranked[-1].score


def test_model_action_competes_instead_of_becoming_command():
    modulators = ModulatorState()
    selector = ActionSelector()
    model = selector.from_model_actions(
        [{
            "kind": "library",
            "description": "Read a relevant item",
            "utility": 0.8,
            "support": 0.7,
            "information_gain": 0.9,
            "payload": {"action": "read", "item_id": "x"},
        }],
        source="substrate:test",
    )
    selected, ranked = selector.select(
        selector.baseline_candidates(modulators, memory_count=5, workspace_activation=0.4) + model,
        modulators,
    )
    assert any(item.kind == "library" for item in ranked)
    assert all(hasattr(item, "score") for item in ranked)


def test_incomplete_experiment_never_enters_competition():
    selector = ActionSelector()
    actions = selector.from_model_actions(
        [{
            "kind": "experiment",
            "description": "Probe a provider",
            "support": 1.0,
            "utility": 1.0,
            "payload": {},
        }],
        source="substrate:test",
    )
    assert actions == []


def test_executable_experiment_is_admitted():
    selector = ActionSelector()
    actions = selector.from_model_actions(
        [{
            "kind": "experiment",
            "description": "Probe a provider",
            "payload": {
                "experiment_type": "provider_probe",
                "experiment_target": "groq",
                "hypothesis": "the configured provider responds",
            },
        }],
        source="substrate:test",
    )
    assert len(actions) == 1
    assert actions[0].payload["experiment_type"] == "provider_probe"


def test_incomplete_library_action_is_rejected():
    selector = ActionSelector()
    actions = selector.from_model_actions(
        [{
            "kind": "library",
            "description": "Read something",
            "payload": {"action": "read"},
        }],
        source="substrate:test",
    )
    assert actions == []
