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


class _AblationResult:
    def __init__(self, text):
        self.provider = "gemini"
        self.text = text


class _AblationRouter:
    def __init__(self):
        self.prompts = []
        self._responses = [
            _AblationResult(json.dumps({
                "hypothesis": "Persistent questions can influence later planning choices.",
                "rationale": "They remain available across generations."
            })),
            _AblationResult(json.dumps({
                "hypothesis": "Concept hierarchies can combine persistent questions with memory abstractions to diversify later planning choices.",
                "rationale": "Reusable concepts provide additional relations for forming hypotheses."
            })),
        ]

    def generate_with(self, provider, prompt):
        assert provider == "gemini"
        self.prompts.append(prompt)
        return self._responses[len(self.prompts) - 1]


def test_hypothesis_ablation_compares_with_and_without_concepts(monkeypatch):
    def records(name, limit=6):
        if name == "concepts.jsonl":
            return [{
                "name": "Memory abstraction",
                "summary": "Reusable concepts connect evidence across generations."
            }]
        return []

    monkeypatch.setattr(experiments, "recent_memory_records", records)
    monkeypatch.setattr(
        experiments,
        "read_json",
        lambda name, default: {
            "question": "Do conceptual memories change future hypothesis generation?"
        } if name == "attention.json" else default,
    )

    router = _AblationRouter()
    out = experiments.run_experiment(
        "hypothesis_ablation",
        "gemini",
        router,
        "Concept memory changes hypothesis generation.",
    )

    assert out["passed"] is True
    assert out["provider"] == "gemini"
    assert len(router.prompts) == 2
    assert "Concepts:" not in router.prompts[0]
    assert "Concepts:" in router.prompts[1]
    assert out["metrics"]["lexical_divergence"] > 0
    assert out["metrics"]["with_concepts_concept_overlap"] >= out["metrics"]["baseline_concept_overlap"]
    assert "not a general causal estimate" in out["identification"]
