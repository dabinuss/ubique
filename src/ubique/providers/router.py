from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from .base import Provider, ProviderError
from ..models import ProviderResult
from ..state import read_json, write_json


class ProviderRouter:
    def __init__(self, providers: Iterable[Provider]):
        self.providers = list(providers)
        self.ledger = read_json("providers.json", {})

    def _record(self, name: str) -> dict:
        return self.ledger.setdefault(
            name, {"successes": 0, "failures": 0, "disabled_until": None}
        )

    def _disabled(self, name: str) -> bool:
        rec = self._record(name)
        value = rec.get("disabled_until")
        if not value:
            return False
        try:
            return datetime.fromisoformat(value) > datetime.now(timezone.utc)
        except ValueError:
            return False

    def _success(self, name: str) -> None:
        rec = self._record(name)
        rec["successes"] = int(rec.get("successes", 0)) + 1
        rec["disabled_until"] = None
        write_json("providers.json", self.ledger)

    def _failure(self, name: str) -> None:
        rec = self._record(name)
        rec["failures"] = int(rec.get("failures", 0)) + 1
        if name != "fallback":
            rec["disabled_until"] = (
                datetime.now(timezone.utc) + timedelta(hours=6)
            ).isoformat()
        write_json("providers.json", self.ledger)

    def generate(self, prompt: str) -> ProviderResult:
        errors: list[str] = []

        for provider in self.providers:
            if not provider.available() or self._disabled(provider.name):
                continue
            try:
                result = provider.generate(prompt)
                self._success(provider.name)
                return result
            except Exception as exc:
                self._failure(provider.name)
                errors.append(f"{provider.name}: {exc}")

        raise ProviderError("No provider succeeded: " + "; ".join(errors))
