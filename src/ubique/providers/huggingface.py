from __future__ import annotations

import httpx
from .base import Provider, ProviderError
from ..models import ProviderResult


class HuggingFaceProvider(Provider):
    name = "huggingface"

    def __init__(self, token: str, model: str, endpoint: str):
        self.token = token
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def available(self) -> bool:
        return bool(self.token)

    def generate(self, prompt: str) -> ProviderResult:
        if not self.available():
            raise ProviderError("Hugging Face token not configured")

        url = f"{self.endpoint}/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 3000,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }

        with httpx.Client(timeout=90) as client:
            r = client.post(url, headers=headers, json=payload)

        if r.status_code >= 400:
            raise ProviderError(f"Hugging Face HTTP {r.status_code}: {r.text[:300]}")

        data = r.json()
        text = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            text = data[0].get("generated_text")
        elif isinstance(data, dict):
            text = data.get("generated_text")

        if not text:
            raise ProviderError(f"Unexpected Hugging Face response: {str(data)[:400]}")

        return ProviderResult(provider=self.name, model=self.model, text=str(text).strip())
