from ubique.models import Task
from ubique.planner import parse_task


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
