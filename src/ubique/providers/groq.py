from __future__ import annotations

import httpx

from .base import Provider, ProviderError
from ..models import ProviderResult


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> ProviderResult:
        if not self.available():
            raise ProviderError("Groq API key not configured")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_completion_tokens": 4096,
        }

        with httpx.Client(timeout=60) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise ProviderError(
                f"Groq HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected Groq response: {str(data)[:400]}"
            ) from exc

        return ProviderResult(
            provider=self.name,
            model=self.model,
            text=str(text).strip(),
        )
