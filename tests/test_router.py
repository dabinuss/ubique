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
