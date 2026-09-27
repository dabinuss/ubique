import json

import ubique.resolution as resolution


def test_parse_resolution_requires_grounded_status():
    out = resolution.parse_resolution(json.dumps({
        "status": "unresolved",
        "answer": "The current evidence does not answer the question.",
        "evidence_basis": "The recorded experiment measured lexical divergence only.",
        "remaining_unknowns": "AUC-ROC was not measured.",
        "confidence": 0.9,
    }))
    assert out["status"] == "unresolved"
    assert out["confidence"] == 0.9


def test_persist_resolution_closes_answered_project(monkeypatch):
    attention = {
        "project_id": "project:1",
        "question": "Does X improve Y?",
        "hypothesis": "X improves Y.",
    }
    projects = {"projects": [{"id": "project:1", "status": "active", "step": 2}]}
    def fake_read(name, default):
        if name == "attention.json":
            return attention.copy()
        if name == "projects.json":
            return projects
        return default
    writes = {}
    records = []
    monkeypatch.setattr(resolution, "read_json", fake_read)
    monkeypatch.setattr(resolution, "write_json", lambda name, value: writes.__setitem__(name, value))
    monkeypatch.setattr(resolution, "append_memory_record", lambda name, value, limit=500: records.append((name, value)))
    monkeypatch.setattr(resolution, "utc_now", lambda: "now")

    result = resolution.persist_resolution(7, {
        "status": "supported",
        "answer": "X improved Y in the observed test.",
        "evidence_basis": "Observed test result.",
        "remaining_unknowns": "Generalization remains unknown.",
        "confidence": 0.8,
    })

    assert records[0][0] == "knowledge.jsonl"
    assert writes["projects.json"]["projects"][0]["status"] == "completed"
    assert result["attention"]["next_command"] == "reflect"


def test_unresolved_resolution_keeps_project_open(monkeypatch):
    attention = {"project_id": "project:1", "question": "Q", "hypothesis": "H"}
    projects = {"projects": [{"id": "project:1", "status": "active", "step": 2}]}
    monkeypatch.setattr(
        resolution,
        "read_json",
        lambda name, default: attention.copy() if name == "attention.json" else projects if name == "projects.json" else default,
    )
    writes = {}
    monkeypatch.setattr(resolution, "write_json", lambda name, value: writes.__setitem__(name, value))
    monkeypatch.setattr(resolution, "append_memory_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(resolution, "utc_now", lambda: "now")

    resolution.persist_resolution(8, {
        "status": "unresolved",
        "answer": "Not enough evidence.",
        "evidence_basis": "Only lexical metrics were observed.",
        "remaining_unknowns": "Task accuracy is missing.",
        "confidence": 0.7,
    })

    assert writes["projects.json"]["projects"][0]["status"] == "blocked"
    assert writes["attention.json"]["next_command"] == "reflect"


def test_conservative_resolution_never_invents_support():
    out = resolution.conservative_resolution_from_attention({
        "question": "Did AUC improve?",
        "hypothesis": "AUC improved by 0.03.",
        "experiment_type": "hypothesis_ablation",
        "last_evidence": '{"kind":"hypothesis_ablation","metrics":{"lexical_divergence":0.4}}',
    })
    assert out["status"] == "unresolved"
    assert "does not establish" in out["answer"]
    assert "lexical_divergence" in out["evidence_basis"]
