import json
import pytest

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
