from __future__ import annotations

import re
from .base import Provider
from ..models import ProviderResult


class FallbackProvider(Provider):
    name = "fallback"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str) -> ProviderResult:
        task = prompt.split("Task:\n", 1)[-1].strip()
        clean = re.sub(r"\s+", " ", task)

        if len(clean) > 900:
            clean = clean[:897] + "..."

        text = (
            "No remote inference provider was available for this cycle.\n\n"
            "Task has been preserved for deterministic handling / later retry:\n\n"
            f"{clean or '(empty task)'}"
        )
        return ProviderResult(provider=self.name, model="deterministic", text=text)
