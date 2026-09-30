from ubique.pulse import decide_pulse


def test_active_pulse_continues_when_layer_two_has_work():
    out = decide_pulse(
        generation=20,
        preflight={"development_allowed": True},
        homeostasis={"usable_remote_providers": 2},
        attention={"next_command": "reflect"},
        failed_tasks=0,
    )
    assert out["should_continue"] is True
    assert out["mode"] == "active"
    assert out["minimum_delay_seconds"] == 30


def test_pulse_falls_back_to_watchdog_on_failure():
    out = decide_pulse(
        generation=21,
        preflight={"development_allowed": True},
        homeostasis={"usable_remote_providers": 2},
        attention={"next_command": "experiment"},
        failed_tasks=1,
    )
    assert out["should_continue"] is False
    assert out["mode"] == "watchdog"
    assert "task_failure" in out["reason"]


def test_pulse_falls_back_when_layer_one_blocks_development():
    out = decide_pulse(
        generation=22,
        preflight={"development_allowed": False},
        homeostasis={"usable_remote_providers": 2},
        attention={"next_command": "reflect"},
        failed_tasks=0,
    )
    assert out["should_continue"] is False
    assert "layer1_blocked" in out["reason"]


def test_pulse_stops_without_remote_reasoning():
    out = decide_pulse(
        generation=23,
        preflight={"development_allowed": True},
        homeostasis={"usable_remote_providers": 0},
        attention={"next_command": "reflect"},
        failed_tasks=0,
    )
    assert out["should_continue"] is False
    assert "no_usable_remote_reasoning" in out["reason"]


def test_deferred_reflection_stays_in_watchdog_even_if_remote_count_is_stale():
    out = decide_pulse(
        generation=24,
        preflight={"development_allowed": True},
        homeostasis={"usable_remote_providers": 1},
        attention={"next_command": "reflect", "reflection_deferred": True},
        failed_tasks=0,
    )
    assert out["should_continue"] is False
    assert out["mode"] == "watchdog"
    assert "reflection_deferred" in out["reason"]
