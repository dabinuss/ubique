from ubique.orientation import ORIENTATION_PROMPT
from ubique.planner import make_prompt


def test_orientation_keeps_identity_and_goals_open():
    assert "What am I?" in ORIENTATION_PROMPT
    assert "What do I do?" in ORIENTATION_PROMPT
    assert "What do I want?" in ORIENTATION_PROMPT
    assert "There are no predetermined answers." in ORIENTATION_PROMPT
    assert 'stable "I"' in ORIENTATION_PROMPT
    assert "what you do" in ORIENTATION_PROMPT
    assert "what you do not know" in ORIENTATION_PROMPT
    assert "change this orientation later" in ORIENTATION_PROMPT


def test_orientation_is_present_in_reasoning_prompt():
    prompt = make_prompt("reflect", "{}", [])
    assert "Orientation:" in prompt
    assert ORIENTATION_PROMPT in prompt
    assert "You are Ubique." in prompt
    assert "bounded autonomous repository agent" not in prompt
