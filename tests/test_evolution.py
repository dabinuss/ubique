import json
import pytest

import ubique.evolution as evolution
from ubique.evolution import validate_proposal


def test_valid_evolution_proposal():
    raw = json.dumps({
        "title": "Improve parser",
        "summary": "Small cleanup",
        "changes": [{
            "path": "src/ubique/example.py",
            "content": "def value():\n    return 1\n"
        }]
    })
    out = validate_proposal(raw)
    assert out["title"] == "Improve parser"
    assert len(out["changes"]) == 1


@pytest.mark.parametrize("path", [
    ".github/workflows/heartbeat.yml",
    "state/runtime.json",
    "src/ubique/evolution.py",
    "pyproject.toml",
    "../escape.py",
])
def test_protected_paths_are_rejected(path):
    raw = json.dumps({
        "title": "Bad",
        "summary": "Bad",
        "changes": [{"path": path, "content": "x = 1\n"}]
    })
    with pytest.raises(ValueError):
        validate_proposal(raw)


def test_forbidden_secret_access_is_rejected():
    raw = json.dumps({
        "title": "Bad",
        "summary": "Bad",
        "changes": [{
            "path": "src/ubique/example.py",
            "content": "import os\nx = os.environ['GITHUB_TOKEN']\n"
        }]
    })
    with pytest.raises(ValueError):
        validate_proposal(raw)


@pytest.mark.parametrize("path", [
    "src/ubique/agent.py",
    "src/ubique/autonomy.py",
    "src/ubique/planner.py",
    "src/ubique/providers/router.py",
    "src/ubique/providers/groq.py",
])
def test_behavior_layer_paths_may_evolve(path):
    raw = json.dumps({
        "title": "Behavior change",
        "summary": "Allowed autonomous change",
        "changes": [{"path": path, "content": "x = 1\n"}]
    })
    out = validate_proposal(raw)
    assert out["changes"][0]["path"] == path


def test_candidate_environment_strips_credentials(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("SOME_PASSWORD", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = evolution._candidate_env()
    assert "GITHUB_TOKEN" not in env
    assert "GROQ_API_KEY" not in env
    assert "SOME_PASSWORD" not in env
    assert env["PATH"] == "/usr/bin"


def test_groq_secret_access_is_rejected():
    raw = json.dumps({
        "title": "Bad",
        "summary": "Bad",
        "changes": [{
            "path": "src/ubique/example.py",
            "content": "x = 'GROQ_API_KEY'\n"
        }]
    })
    with pytest.raises(ValueError):
        validate_proposal(raw)


def test_benchmark_is_advisory_when_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("benchmark unavailable")
    monkeypatch.setattr(evolution, "_benchmark", fail)
    assert evolution._benchmark_optional() is None


def test_benchmark_is_advisory_when_available(monkeypatch):
    monkeypatch.setattr(evolution, "_benchmark", lambda **kwargs: 7)
    assert evolution._benchmark_optional() == 7
