from ubique.agent import make_evolution_repair_prompt


def test_repair_prompt_contains_failure_and_original():
    prompt = make_evolution_repair_prompt(
        '{"title":"x","changes":[]}',
        "candidate tests failed: SyntaxError: invalid syntax",
    )
    assert "SyntaxError" in prompt
    assert '"title":"x"' in prompt
    assert "Return ONE corrected proposal as strict JSON only" in prompt


def test_repair_prompt_reasserts_protected_core():
    prompt = make_evolution_repair_prompt("{}", "proposal rejected: bad")
    assert "protected recovery/evolution kernel" in prompt
    assert "Philosophical orientation and optional theories are not protected doctrine" in prompt
    assert "workflows" in prompt
    assert "JSON only" in prompt
