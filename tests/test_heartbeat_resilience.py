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
