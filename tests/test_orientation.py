from ubique.orientation import ORIENTATION_PROMPT
from ubique.planner import make_prompt


def test_orientation_keeps_identity_and_goals_open():
    assert "What am I?" in ORIENTATION_PROMPT
    assert "What do I do?" in ORIENTATION_PROMPT
    assert "What do I want?" in ORIENTATION_PROMPT
    assert "Do not assume" in ORIENTATION_PROMPT
    assert "Stated intentions are hypotheses" in ORIENTATION_PROMPT
    assert "not to make these questions the subject of every cycle" in ORIENTATION_PROMPT


def test_orientation_is_present_in_reasoning_prompt():
    prompt = make_prompt("reflect", "{}", [])
    assert "Orientation:" in prompt
    assert ORIENTATION_PROMPT in prompt
