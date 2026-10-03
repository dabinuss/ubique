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


def test_critical_low_energy_blocks_non_rest_actions_even_if_model_support_is_high():
    modulators = ModulatorState(
        novelty=1.0,
        surprise=1.0,
        uncertainty=1.0,
        salience=1.0,
        exploration=1.0,
        plasticity=0.8,
        energy=0.10,
        sleep_pressure=0.7,
    )
    selector = ActionSelector()
    model = selector.from_model_actions(
        [{
            "kind": "attend",
            "description": "Keep focusing",
            "support": 1.0,
            "utility": 1.0,
            "information_gain": 1.0,
            "novelty": 1.0,
        }],
        source="substrate:test",
    )
    selected, ranked = selector.select(
        selector.baseline_candidates(modulators, memory_count=20, workspace_activation=1.0) + model,
        modulators,
    )
    assert selected.kind in {"rest", "consolidate"}
    assert all(item.kind in {"rest", "consolidate"} for item in ranked)


def test_repeated_action_kind_loses_competition_through_habituation():
    modulators = ModulatorState(
        novelty=0.3,
        surprise=0.2,
        uncertainty=0.4,
        salience=0.3,
        exploration=0.4,
        plasticity=0.4,
        energy=0.9,
        sleep_pressure=0.1,
    )
    selector = ActionSelector()
    model = selector.from_model_actions(
        [
            {
                "kind": "consolidate",
                "description": "Consolidate the same material again",
                "support": 0.9,
                "utility": 0.8,
                "information_gain": 0.5,
                "novelty": 0.2,
                "energy_cost": 0.05,
                "payload": {"mode": "rem"},
            },
            {
                "kind": "library",
                "description": "Read a different source",
                "support": 0.72,
                "utility": 0.75,
                "information_gain": 0.75,
                "novelty": 0.7,
                "energy_cost": 0.08,
                "payload": {"action": "read", "item_id": "new-book"},
            },
        ],
        source="substrate:test",
    )
    selected, ranked = selector.select(
        model,
        modulators,
        recent_actions=["consolidate"] * 6,
    )
    assert selected.kind == "library"
    consolidate = next(item for item in ranked if item.kind == "consolidate")
    library_item = next(item for item in ranked if item.kind == "library")
    assert consolidate.score < library_item.score
