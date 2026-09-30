from ubique.models import Task
from ubique.planner import make_prompt, parse_task


def test_parse_plan():
    p = parse_task(Task(id="1", title="x", body="/plan\nBuild a thing"))
    assert p.command == "plan"
    assert p.payload == "Build a thing"


def test_unknown_defaults_to_think():
    p = parse_task(Task(id="1", title="Hello", body="World"))
    assert p.command == "think"
    assert "Hello" in p.payload and "World" in p.payload


def test_empty_body_uses_title():
    p = parse_task(Task(id="1", title="Do this", body=""))
    assert p.command == "think"
    assert p.payload == "Do this"


def test_fzg_is_not_injected_as_normative_system_basis():
    prompt = make_prompt("reflect", "{}", [])
    assert "FZG v1.0 is the normative theoretical basis" not in prompt
    assert "Optional library catalog" in prompt
    assert "Library entries are optional sources" in prompt


def test_fzg_command_is_no_longer_first_class():
    p = parse_task(Task(id="1", title="old", body="/fzg\nAnalyze this"))
    assert p.command == "think"


def test_reflect_prompt_removes_model_observation_and_project_fields():
    prompt = make_prompt("reflect", "{}", [])
    assert "Do not output an observation field" in prompt
    assert "project_title" in prompt
    assert "project_objective" in prompt
    assert '"observation"' not in prompt.split("Schema:", 1)[1]
    assert '"project_title"' not in prompt.split("Schema:", 1)[1]


def test_reflect_prompt_does_not_replay_old_reflection_results():
    prompt = make_prompt(
        "reflect",
        "{}",
        [{"command": "reflect", "result": "invented hidden-state observation"}],
    )
    assert "invented hidden-state observation" not in prompt
    assert "intentionally omitted" in prompt
