from ubique.autonomy import autonomous_task


def test_autonomy_uses_fzg_without_remote_provider():
    task = autonomous_task(3, remote_reasoning_available=False)
    assert task.source == "autonomous"
    assert task.body.startswith("/fzg")


def test_autonomy_evolves_every_third_generation_with_provider():
    task = autonomous_task(3, remote_reasoning_available=True)
    assert task.body.startswith("/evolve")
    assert "FZG v1.0" in task.body


def test_autonomy_observes_between_evolution_cycles():
    task = autonomous_task(4, remote_reasoning_available=True)
    assert task.body.startswith("/fzg")
