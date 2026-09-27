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
