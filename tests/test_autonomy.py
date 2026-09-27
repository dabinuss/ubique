from ubique.autonomy import autonomous_task


def test_without_remote_provider_uses_cheap_status_cycle():
    task = autonomous_task(6, remote_reasoning_available=False)
    assert task.source == "autonomous"
    assert task.body.startswith("/status")


def test_autonomy_evolves_every_24th_generation_with_provider():
    task = autonomous_task(24, remote_reasoning_available=True)
    assert task.body.startswith("/evolve")
    assert "FZG v1.0" in task.body


def test_autonomy_uses_fzg_every_6th_generation_with_provider():
    task = autonomous_task(6, remote_reasoning_available=True)
    assert task.body.startswith("/fzg")


def test_autonomy_uses_status_between_remote_cycles():
    task = autonomous_task(7, remote_reasoning_available=True)
    assert task.body.startswith("/status")
