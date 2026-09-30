import json

import ubique.cognition as cognition


def test_parse_reflection_allows_philosophy_without_experiment():
    raw = json.dumps({
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


def test_model_cannot_write_observation_or_project_identity():
    raw = json.dumps({
        "observation": "I secretly measured a hidden state.",
        "project_title": "Autonomous Epistemic Question Generation & Memory Abstraction",
        "project_objective": "Turn reflection into a technical project.",
        "question": "What do I do?",
        "reflection": "Interpret what is actually recorded.",
        "provisional_answer": "I transform context into responses and actions.",
        "uncertainty": "This remains provisional.",
        "next_action": "Continue reflecting.",
        "next_command": "reflect",
    })
    out = cognition.parse_reflection(raw)
    assert "observation" not in out
    assert "project_title" not in out
    assert "project_objective" not in out


def test_parse_reflection_requires_experiment_fields_only_for_experiment():
    raw = json.dumps({
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


def test_persist_reflection_uses_system_facts_and_fixed_focus(monkeypatch):
    monkeypatch.setattr(cognition, "read_json", lambda name, default: default)
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    records = []
    monkeypatch.setattr(
        cognition,
        "append_memory_record",
        lambda name, value, limit=500: records.append((name, value)),
    )
    reflection = {
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
        "confidence": 0.4,
        "importance": 0.8,
    }
    facts = [{"generation": 12, "command": "experiment", "success": True}]
    attention = cognition.persist_reflection(13, reflection, facts)
    assert [name for name, _ in records] == ["thoughts.jsonl"]
    thought = records[0][1]
    assert thought["observed_facts"] == facts
    assert thought["observation_status"] == "recorded_system_facts"
    assert "generation 12: experiment succeeded" in thought["observation"]
    assert "projects.json" not in writes
    assert "project_id" not in attention
    assert attention["focus"] == "Philosophical self-inquiry"
    assert attention["observed_facts"] == facts


def test_executed_action_facts_do_not_replay_reflection_prose(monkeypatch):
    monkeypatch.setattr(
        cognition,
        "recent_episodes",
        lambda limit=30: [
            {
                "generation": 1,
                "command": "reflect",
                "success": True,
                "result": "invented hidden-state observation",
            },
            {
                "generation": 2,
                "command": "experiment",
                "success": True,
                "result": "measured experiment outcome",
            },
        ],
    )
    facts = cognition._executed_action_facts()
    reflect_fact = next(x for x in facts if x["command"] == "reflect")
    experiment_fact = next(x for x in facts if x["command"] == "experiment")
    assert "recorded_result" not in reflect_fact
    assert experiment_fact["recorded_result"] == "measured experiment outcome"


def test_stagnation_counts_status_and_resets_on_reflection(monkeypatch):
    stored = {"level": 2, "last_command": "status"}
    monkeypatch.setattr(cognition, "read_json", lambda name, default: stored.copy())
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    assert cognition.update_stagnation("status", 4)["level"] == 3
    assert cognition.update_stagnation("reflect", 5)["level"] == 0


def test_two_nearly_identical_consecutive_answers_trigger_stagnation():
    thoughts = [
        {
            "id": "thought:1",
            "question": "What do I do?",
            "provisional_answer": "I engage in self directed epistemic cycles that generate questions hypotheses and reflections.",
        },
        {
            "id": "thought:2",
            "question": "What do I do?",
            "provisional_answer": "I engage in self directed epistemic cycles by generating questions hypotheses and reflective answers.",
        },
    ]
    out = cognition.assess_reflection_stagnation(thoughts)
    assert out["detected"] is True
    assert out["reason"] == "repeated_consecutive_answer"


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


def test_status_result_is_not_replayed_as_fact_body(monkeypatch):
    monkeypatch.setattr(
        cognition,
        "recent_episodes",
        lambda limit=30: [
            {
                "generation": 3,
                "task_id": "issue:1",
                "command": "status",
                "success": True,
                "result": "old attention text that must not become evidence",
            }
        ],
    )
    facts = cognition._executed_action_facts()
    assert facts[0]["fact_id"] == "episode:3:issue:1"
    assert "recorded_result" not in facts[0]


def test_unbacked_inference_is_downgraded_to_hypothesis():
    facts = [
        {
            "fact_id": "episode:5:autonomous:experiment:5",
            "generation": 5,
            "command": "experiment",
            "success": True,
        }
    ]
    claims = [
        {
            "kind": "inference",
            "statement": "I have a stable hidden identity.",
            "basis_fact_ids": ["not-a-real-fact"],
        },
        {
            "kind": "inference",
            "statement": "An experiment action succeeded.",
            "basis_fact_ids": ["episode:5:autonomous:experiment:5"],
        },
    ]
    out = cognition._normalize_claims(claims, facts)
    assert out[0]["kind"] == "hypothesis"
    assert out[0]["downgraded_from"] == "inference"
    assert out[1]["kind"] == "inference"
    assert out[1]["epistemic_status"] == "grounded_inference"


def test_defer_reflection_preserves_previous_thought_and_marks_retry(monkeypatch):
    previous = {
        "question": "What am I?",
        "reflection": "Previous interpretation.",
        "next_command": "reflect",
    }
    monkeypatch.setattr(cognition, "read_json", lambda name, default: previous.copy())
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    out = cognition.defer_reflection(20, "remote unavailable")
    assert out["reflection"] == "Previous interpretation."
    assert out["reflection_deferred"] is True
    assert out["next_command"] == "reflect"
    assert writes["attention.json"]["deferred_reason"] == "remote unavailable"


def test_persisted_claims_are_epistemically_typed(monkeypatch):
    monkeypatch.setattr(cognition, "read_json", lambda name, default: default)
    writes = {}
    monkeypatch.setattr(cognition, "write_json", lambda name, value: writes.__setitem__(name, value))
    records = []
    monkeypatch.setattr(
        cognition,
        "append_memory_record",
        lambda name, value, limit=500: records.append((name, value)),
    )
    reflection = {
        "question": "What am I?",
        "reflection": "One interpretation is possible.",
        "claims": [{
            "kind": "inference",
            "statement": "A hidden persistent self exists.",
            "basis_fact_ids": [],
        }],
        "provisional_answer": "The answer remains provisional.",
        "uncertainty": "High.",
        "hypothesis": "",
        "proposed_experiment": "",
        "expected_evidence": "",
        "next_action": "Reflect further.",
        "next_command": "reflect",
        "experiment_type": "",
        "experiment_target": "",
        "confidence": 0.2,
        "importance": 0.8,
    }
    attention = cognition.persist_reflection(21, reflection, [])
    assert attention["epistemic_schema_version"] == 2
    assert attention["claims"][0]["kind"] == "hypothesis"
    assert records[0][1]["epistemic_schema_version"] == 2
