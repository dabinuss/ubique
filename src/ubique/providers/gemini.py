from __future__ import annotations

import httpx
from .base import Provider, ProviderError
from ..models import ProviderResult


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> ProviderResult:
        if not self.available():
            raise ProviderError("Gemini API key not configured")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }

        with httpx.Client(timeout=60) as client:
            r = client.post(url, params={"key": self.api_key}, json=payload)

        if r.status_code >= 400:
            raise ProviderError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")

        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected Gemini response: {str(data)[:400]}") from exc

        return ProviderResult(provider=self.name, model=self.model, text=text.strip())
