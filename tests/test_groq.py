from ubique.providers.groq import GroqProvider


def test_groq_unavailable_without_key():
    provider = GroqProvider("", "openai/gpt-oss-120b")
    assert provider.available() is False


def test_groq_available_with_key():
    provider = GroqProvider("test-key", "openai/gpt-oss-120b")
    assert provider.available() is True
