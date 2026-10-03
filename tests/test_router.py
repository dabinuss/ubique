import ubique.state as state
import ubique.providers.router as router_mod
from ubique.providers.base import Provider, ProviderError
from ubique.models import ProviderResult


class Bad(Provider):
    name = "bad"
    def available(self): return True
    def generate(self, prompt): raise ProviderError("boom")


class Good(Provider):
    name = "good"
    def available(self): return True
    def generate(self, prompt): return ProviderResult(provider="good", text="ok")


def test_router_falls_through(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(router_mod, "read_json", lambda name, default: {})
    written = {}
    monkeypatch.setattr(router_mod, "write_json", lambda name, value: written.update(value))

    r = router_mod.ProviderRouter([Bad(), Good()])
    out = r.generate("x")
    assert out.provider == "good"
    assert written["bad"]["failures"] == 1
    assert written["good"]["successes"] == 1


class NamedGood(Provider):
    def __init__(self, name):
        self.name = name
    def available(self): return True
    def generate(self, prompt): return ProviderResult(provider=self.name, text="ok")


def test_select_remote_skips_exhausted_preferred(monkeypatch):
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    monkeypatch.setattr(router_mod, "read_json", lambda name, default: {
        "gemini": {"daily_date": today, "daily_calls": 20},
        "groq": {"daily_date": today, "daily_calls": 5},
    })
    monkeypatch.setattr(router_mod, "write_json", lambda name, value: None)
    r = router_mod.ProviderRouter(
        [NamedGood("gemini"), NamedGood("groq")],
        daily_limits={"gemini": 20, "groq": 1000},
    )
    assert r.select_remote("gemini", required_calls=2) == "groq"


def test_remote_eligibility_reports_exhausted_budget(monkeypatch):
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    monkeypatch.setattr(router_mod, "read_json", lambda name, default: {
        "gemini": {"daily_date": today, "daily_calls": 20},
        "groq": {"daily_date": today, "daily_calls": 7},
    })
    monkeypatch.setattr(router_mod, "write_json", lambda name, value: None)
    r = router_mod.ProviderRouter(
        [NamedGood("gemini"), NamedGood("groq")],
        daily_limits={"gemini": 20, "groq": 1000},
    )
    status = r.remote_eligibility()
    assert status["gemini"]["eligible"] is False
    assert status["gemini"]["remaining_calls"] == 0
    assert status["groq"]["eligible"] is True


def test_transient_provider_failures_use_short_adaptive_cooldown(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(router_mod, "read_json", lambda name, default: {})
    monkeypatch.setattr(router_mod, "write_json", lambda name, value: None)
    r = router_mod.ProviderRouter([NamedGood("groq")])

    before = datetime.now(timezone.utc)
    r._failure("groq", ProviderError("temporary network timeout"))
    first = r.ledger["groq"]
    first_until = datetime.fromisoformat(first["disabled_until"])
    assert first["consecutive_failures"] == 1
    assert 4 * 60 <= (first_until - before).total_seconds() <= 6 * 60

    r._failure("groq", ProviderError("temporary network timeout"))
    second = r.ledger["groq"]
    second_until = datetime.fromisoformat(second["disabled_until"])
    assert second["consecutive_failures"] == 2
    assert 9 * 60 <= (second_until - datetime.now(timezone.utc)).total_seconds() <= 11 * 60

    r._success("groq")
    assert r.ledger["groq"]["consecutive_failures"] == 0
    assert r.ledger["groq"]["disabled_until"] is None


def test_auth_failure_keeps_long_cooldown(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(router_mod, "read_json", lambda name, default: {})
    monkeypatch.setattr(router_mod, "write_json", lambda name, value: None)
    r = router_mod.ProviderRouter([NamedGood("groq")])

    before = datetime.now(timezone.utc)
    r._failure("groq", ProviderError("Groq HTTP 401: invalid API key"))
    until = datetime.fromisoformat(r.ledger["groq"]["disabled_until"])
    assert 5.9 * 3600 <= (until - before).total_seconds() <= 6.1 * 3600
