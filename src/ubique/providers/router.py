from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from .base import Provider, ProviderError
from ..models import ProviderResult
from ..state import read_json, write_json


class ProviderRouter:
    def __init__(
        self,
        providers: Iterable[Provider],
        daily_limits: dict[str, int] | None = None,
    ):
        self.providers = list(providers)
        self.daily_limits = daily_limits or {}
        self.ledger = read_json("providers.json", {})

    def _record(self, name: str) -> dict:
        return self.ledger.setdefault(
            name,
            {
                "successes": 0,
                "failures": 0,
                "disabled_until": None,
                "daily_date": None,
                "daily_calls": 0,
            },
        )

    def _refresh_daily(self, name: str) -> dict:
        rec = self._record(name)
        today = datetime.now(timezone.utc).date().isoformat()
        if rec.get("daily_date") != today:
            rec["daily_date"] = today
            rec["daily_calls"] = 0
        return rec

    def _daily_budget_available(self, name: str) -> bool:
        limit = self.daily_limits.get(name)
        if limit is None:
            return True
        rec = self._refresh_daily(name)
        return int(rec.get("daily_calls", 0)) < max(0, int(limit))

    def _consume_call(self, name: str) -> None:
        rec = self._refresh_daily(name)
        rec["daily_calls"] = int(rec.get("daily_calls", 0)) + 1
        write_json("providers.json", self.ledger)

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
            if not self._daily_budget_available(provider.name):
                errors.append(f"{provider.name}: daily request budget exhausted")
                continue
            try:
                if provider.name != "fallback":
                    # Count the attempt before sending it. Failed requests still
                    # consume quota, so this is deliberately conservative.
                    self._consume_call(provider.name)
                result = provider.generate(prompt)
                self._success(provider.name)
                return result
            except Exception as exc:
                self._failure(provider.name)
                errors.append(f"{provider.name}: {exc}")

        raise ProviderError("No provider succeeded: " + "; ".join(errors))


    def generate_with(self, provider_name: str, prompt: str) -> ProviderResult:
        """Run one bounded experiment through a specifically named configured provider."""
        provider = next((p for p in self.providers if p.name == provider_name), None)
        if provider is None or provider.name == "fallback":
            raise ProviderError(f"Provider is not eligible for a remote probe: {provider_name}")
        if not provider.available():
            raise ProviderError(f"Provider is not configured: {provider_name}")
        if self._disabled(provider.name):
            raise ProviderError(f"Provider is temporarily disabled: {provider_name}")
        if not self._daily_budget_available(provider.name):
            raise ProviderError(f"Provider daily request budget exhausted: {provider_name}")

        self._consume_call(provider.name)
        try:
            result = provider.generate(prompt)
            self._success(provider.name)
            return result
        except Exception:
            self._failure(provider.name)
            raise
