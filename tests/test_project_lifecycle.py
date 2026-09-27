import json

import ubique.project_lifecycle as lifecycle


def test_loop_detector_flags_repeated_experiment(monkeypatch):
    monkeypatch.setattr(lifecycle, "read_json", lambda name, default: {
        "projects": [{
            "id": "project:14", "title": "Memory", "status": "active", "step": 12,
            "history": [{"generation": i, "thought": f"thought:{i}"} for i in range(1, 9)]
        }]
    })
    monkeypatch.setattr(lifecycle, "recent_memory_records", lambda name, limit: [
        {
            "id": f"thought:{i}",
            "question": f"memory depth question {i}",
            "hypothesis": "hierarchical memory improves planning",
            "next_action": "run memory abstraction",
            "experiment_type": "memory_abstraction",
        }
        for i in range(1, 9)
    ])
    out = lifecycle.assess_project_loop()
    assert out["detected"] is True
    assert out["dominant_experiment_type"] == "memory_abstraction"
    assert out["dominant_experiment_count"] >= 5


def test_pause_project_persists_summary_and_forces_new_frontier(monkeypatch):
    project_state = {"projects": [{"id": "project:14", "title": "Old", "status": "active"}]}
    writes = {}
    appended = []
    monkeypatch.setattr(lifecycle, "read_json", lambda name, default: project_state)
    monkeypatch.setattr(lifecycle, "write_json", lambda name, value: writes.__setitem__(name, value))
    monkeypatch.setattr(lifecycle, "append_memory_record", lambda name, value, limit=200: appended.append((name, value)))
    review = {
        "learned": ["A"], "not_established": ["B"], "supported_hypotheses": [],
        "weakened_hypotheses": [], "loop_reason": "repetition",
        "resume_when": ["new benchmark"], "handoff_question": "What next?"
    }
    curiosity = {"frontiers": [{"id": "capability_composition", "question": "What can be composed?", "novelty_priority": 1.0}]}
    out = lifecycle.pause_project(50, review, {"project_id": "project:14", "reason": "loop"}, curiosity)
    assert project_state["projects"][0]["status"] == "paused"
    assert appended[0][0] == "project_summaries.jsonl"
    assert out["attention"]["force_new_project"] is True
    assert out["attention"]["forced_frontier"]["id"] == "capability_composition"


def test_project_review_requires_uncertainty_fields():
    data = {
        "learned": ["x"],
        "not_established": ["y"],
        "loop_reason": "repeat",
        "resume_when": ["new evidence"],
        "handoff_question": "new question",
    }
    out = lifecycle.parse_project_review(json.dumps(data))
    assert out["learned"] == ["x"]
    assert out["not_established"] == ["y"]
