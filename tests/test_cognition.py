import json

import ubique.cognition as cognition


def test_parse_reflection_bounds_and_whitelists_next_command():
    raw = json.dumps({
        "observation": "Groq is configured but under-tested.",
        "question": "Is redundancy operational?",
        "hypothesis": "Groq can complete a bounded reasoning task.",
        "proposed_experiment": "Route one bounded task through Groq.",
        "expected_evidence": "A successful provider ledger entry.",
        "next_action": "Validate the second provider path.",
        "next_command": "delete_everything",
        "confidence": 4,
        "importance": -2,
    })
    out = cognition.parse_reflection(raw)
    assert out["next_command"] == "reflect"
    assert out["confidence"] == 1.0
    assert out["importance"] == 0.0


def test_persist_reflection_creates_continuity(tmp_path, monkeypatch):
    monkeypatch.setattr(cognition, "read_json", lambda name, default: default)
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    records = []
    monkeypatch.setattr(cognition, "append_memory_record", lambda name, value, limit=500: records.append((name, value)))
    reflection = {
        "observation": "A",
        "question": "Q",
        "hypothesis": "H",
        "proposed_experiment": "E",
        "expected_evidence": "X",
        "next_action": "N",
        "next_command": "reflect",
        "project_title": "Provider validation",
        "project_objective": "Validate redundancy",
        "confidence": 0.4,
        "importance": 0.8,
    }
    attention = cognition.persist_reflection(13, reflection)
    assert [name for name, _ in records] == ["thoughts.jsonl", "hypotheses.jsonl"]
    assert writes["projects.json"]["projects"][0]["step"] == 1
    assert attention["project_id"] == "project:13"


def test_stagnation_counts_status_and_resets_on_reflection(monkeypatch):
    stored = {"level": 2, "last_command": "status"}
    monkeypatch.setattr(cognition, "read_json", lambda name, default: stored.copy())
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    assert cognition.update_stagnation("status", 4)["level"] == 3
    assert cognition.update_stagnation("reflect", 5)["level"] == 0
