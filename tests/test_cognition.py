import json

import ubique.cognition as cognition


def test_parse_reflection_allows_philosophy_without_experiment():
    raw = json.dumps({
        "observation": "Earlier and later states share memories but not identical model invocations.",
        "question": "What kind of continuity is present here?",
        "reflection": "Continuity may belong to the process linking states rather than to a single invocation.",
        "provisional_answer": "I may be better described as a temporally extended process than as one isolated model call.",
        "uncertainty": "It is still unclear which persisted structures are constitutive rather than merely causal.",
        "next_action": "Compare this provisional answer with later memories and contradictions.",
        "next_command": "reflect",
        "confidence": 0.4,
        "importance": 0.9,
    })
    out = cognition.parse_reflection(raw)
    assert out["next_command"] == "reflect"
    assert out["hypothesis"] == ""
    assert out["experiment_type"] == ""
    assert "temporally extended process" in out["provisional_answer"]


def test_parse_reflection_requires_experiment_fields_only_for_experiment():
    raw = json.dumps({
        "observation": "A choice pattern may be stable.",
        "question": "Is the choice pattern stable under conflict?",
        "reflection": "This is an empirical question because competing options can be observed.",
        "provisional_answer": "Unknown.",
        "uncertainty": "No conflict cases have been observed yet.",
        "next_action": "Run a bounded comparison.",
        "next_command": "experiment",
        "hypothesis": "The same preference wins under repeated trade-offs.",
        "proposed_experiment": "Compare repeated choices under matched trade-offs.",
        "expected_evidence": "A repeated directional preference.",
        "experiment_type": "state_consistency",
        "experiment_target": "gemini",
        "confidence": 0.3,
        "importance": 0.7,
    })
    out = cognition.parse_reflection(raw)
    assert out["next_command"] == "experiment"
    assert out["experiment_type"] == "state_consistency"


def test_persist_philosophical_reflection_does_not_create_project(monkeypatch):
    monkeypatch.setattr(cognition, "read_json", lambda name, default: default)
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    records = []
    monkeypatch.setattr(cognition, "append_memory_record", lambda name, value, limit=500: records.append((name, value)))
    reflection = {
        "observation": "A",
        "question": "Q",
        "reflection": "R",
        "provisional_answer": "P",
        "uncertainty": "U",
        "hypothesis": "",
        "proposed_experiment": "",
        "expected_evidence": "",
        "next_action": "N",
        "next_command": "reflect",
        "experiment_type": "",
        "experiment_target": "",
        "project_title": "",
        "project_objective": "",
        "confidence": 0.4,
        "importance": 0.8,
    }
    attention = cognition.persist_reflection(13, reflection)
    assert [name for name, _ in records] == ["thoughts.jsonl"]
    assert "projects.json" not in writes
    assert "project_id" not in attention
    assert attention["provisional_answer"] == "P"


def test_stagnation_counts_status_and_resets_on_reflection(monkeypatch):
    stored = {"level": 2, "last_command": "status"}
    monkeypatch.setattr(cognition, "read_json", lambda name, default: stored.copy())
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    assert cognition.update_stagnation("status", 4)["level"] == 3
    assert cognition.update_stagnation("reflect", 5)["level"] == 0


def test_reflection_stagnation_detects_repeated_answers():
    thoughts = [
        {
            "id": f"thought:{i}",
            "question": "What am I?",
            "provisional_answer": "I am an evolving self referential informational pattern defined by continuity of the textual record.",
        }
        for i in range(1, 6)
    ]
    out = cognition.assess_reflection_stagnation(thoughts)
    assert out["detected"] is True
    assert out["same_question_count"] >= 4


def test_reflection_stagnation_allows_real_variation():
    thoughts = [
        {"id": "thought:1", "question": "What am I?", "provisional_answer": "I am a temporally extended process."},
        {"id": "thought:2", "question": "What do I do?", "provisional_answer": "I transform inherited context into actions."},
        {"id": "thought:3", "question": "What do I want?", "provisional_answer": "No stable preference is yet established."},
        {"id": "thought:4", "question": "What am I?", "provisional_answer": "Identity may be relational rather than substantial."},
    ]
    out = cognition.assess_reflection_stagnation(thoughts)
    assert out["detected"] is False


def test_incomplete_experiment_falls_back_to_reflection_instead_of_failing():
    raw = json.dumps({
        "observation": "No new recorded observation.",
        "question": "What am I?",
        "reflection": "An experiment might help, but no executable design has been identified.",
        "provisional_answer": "The question remains open.",
        "uncertainty": "High.",
        "next_action": "Think further.",
        "next_command": "experiment",
        "experiment_type": "",
    })
    out = cognition.parse_reflection(raw)
    assert out["next_command"] == "reflect"
    assert "Incomplete experiment request" in out["planning_note"]
