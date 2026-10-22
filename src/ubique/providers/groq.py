from __future__ import annotations

import httpx

from .base import Provider, ProviderError
from ..models import ProviderResult


class GroqProvider(Provider):
    """Provider implementation for the Groq API.

    The provider now accepts an optional *system_prompt* that is sent as a
    ``system``‑role message before the user's prompt.  Supplying a system prompt
    can guide the model to adopt a more curious or reflective stance, which
    aligns with Ubique's goal of continuous learning and belief revision.

    The default behaviour (no system prompt) is unchanged, ensuring backward
    compatibility.
    """

    name = "groq"

    def __init__(self, api_key: str, model: str, system_prompt: str | None = None):
        """Create a new :class:`GroqProvider`.

        Parameters
        ----------
        api_key: str
            The Groq API key.
        model: str
            The model identifier to use (e.g., ``"llama3-8b-8192"``).
        system_prompt: str | None, optional
            An optional system‑role message that will be prepended to the
            conversation.  If ``None`` (the default) the provider behaves exactly
            as before.
        """
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt

    def available(self) -> bool:
        """Return ``True`` when an API key is configured."""
        return bool(self.api_key)

    def generate(self, prompt: str) -> ProviderResult:
        """Generate a completion from Groq.

        The request payload now optionally includes a ``system`` message when
        ``self.system_prompt`` is set.  This small addition encourages the model
        to adopt a more inquisitive tone without affecting existing callers.
        """
        if not self.available():
            raise ProviderError("Groq API key not configured")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build the message list, inserting a system prompt if provided.
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
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
