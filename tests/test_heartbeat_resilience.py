from pathlib import Path


def test_heartbeat_is_offset_from_crowded_quarter_hour():
    workflow = Path(".github/workflows/heartbeat.yml").read_text(encoding="utf-8")
    assert 'cron: "3,18,33,48 * * * *"' in workflow
    assert "closed" in workflow
    assert "unlabeled" in workflow


def test_independent_watchdog_dispatches_when_runtime_is_stale():
    watchdog = Path(".github/workflows/heartbeat-watchdog.yml").read_text(encoding="utf-8")
    assert 'cron: "10,25,40,55 * * * *"' in watchdog
    assert "age_minutes > 22" in watchdog
    assert "heartbeat.yml/dispatches" in watchdog


def test_quiet_heartbeat_self_dispatches_instead_of_waiting_for_cron_only():
    workflow = Path(".github/workflows/heartbeat.yml").read_text(encoding="utf-8")
    assert 'if [ "$SHOULD_CONTINUE" = "true" ]; then' in workflow
    assert "DELAY=780" in workflow
    assert "sleep \"$DELAY\"" in workflow
    assert "heartbeat.yml/dispatches" in workflow
    assert 'Pulse entered watchdog mode; waiting for the scheduled heartbeat.' not in workflow
