import json

import ubique.experiments as experiments


class _Result:
    provider = "gemini"
    text = json.dumps({
        "concepts": [
            {
                "name": "Question persistence",
                "summary": "Questions can guide later action selection across generations.",
                "supports": ["thought:14"],
                "exceptions": ["No evidence yet for long-term stability."],
                "open_question": "Will the concept survive contradictory evidence?",
            }
        ]
    })


class _Router:
    def generate(self, prompt):
        assert "Synthesize reusable conceptual memory" in prompt
        return _Result()


def test_memory_abstraction_persists_concepts(monkeypatch):
    monkeypatch.setattr(
        experiments,
        "recent_memory_records",
        lambda name, limit=8: [{"id": "thought:14", "question": "What is worth pursuing?"}],
    )
    records = []
    monkeypatch.setattr(
        experiments,
        "append_memory_record",
        lambda name, value, limit=300: records.append((name, value)),
    )
    out = experiments.run_experiment("memory_abstraction", "", _Router(), "test hypothesis")
    assert out["passed"] is True
    assert out["concepts_created"] == 1
    assert records[0][0] == "concepts.jsonl"
    assert records[0][1]["name"] == "Question persistence"


def test_unknown_experiment_is_rejected():
    try:
        experiments.run_experiment("arbitrary", "", _Router(), "x")
    except ValueError as exc:
        assert "unsupported experiment" in str(exc)
    else:
        raise AssertionError("unsupported experiment must fail")
