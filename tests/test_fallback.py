from ubique.providers.fallback import FallbackProvider


def test_fallback_always_available():
    assert FallbackProvider().available()


def test_fallback_returns_task():
    r = FallbackProvider().generate("Header\nTask:\nhello world")
    assert r.provider == "fallback"
    assert "hello world" in r.text
