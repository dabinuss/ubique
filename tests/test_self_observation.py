import ubique.self_observation as self_observation


def test_record_self_observation_keeps_raw_transition(monkeypatch):
    records = []
    monkeypatch.setattr(
        self_observation,
        "recent_memory_records",
        lambda name, limit=1: [{"generation": 9}] if limit == 1 else [],
    )
    monkeypatch.setattr(
        self_observation,
        "append_memory_record",
        lambda name, value, limit=500: records.append((name, value, limit)),
    )
    monkeypatch.setattr(self_observation, "utc_now", lambda: "now")

    out = self_observation.record_self_observation(
        generation=10,
        command="experiment",
        provider="groq",
        source="autonomous",
        success=True,
        result='{"passed": true}',
        before_attention={
            "question": "Can I measure X?",
            "hypothesis": "X changes Y.",
            "next_action": "Measure X.",
            "next_command": "experiment",
            "project_id": "project:1",
        },
        after_attention={
            "question": "Can I measure X?",
            "hypothesis": "X changes Y.",
            "next_action": "Interpret evidence.",
            "next_command": "reflect",
            "project_id": "project:1",
        },
    )

    assert out["previous_observation_generation"] == 9
    assert out["action"]["source"] == "autonomous"
    assert out["observed"]["success"] is True
    assert out["state_change"]["next_command"] == {
        "before": "experiment",
        "after": "reflect",
    }
    assert records[0][0] == "self_observations.jsonl"


def test_self_observation_context_is_compact_transition_evidence(monkeypatch):
    monkeypatch.setattr(
        self_observation,
        "recent_memory_records",
        lambda name, limit=8: [{"generation": 3, "action": {"command": "reflect"}}],
    )
    out = self_observation.self_observation_context()
    assert set(out) == {"recent_transitions"}
    assert out["recent_transitions"][0]["generation"] == 3
    assert "meta_questions" not in out
