from __future__ import annotations

from abc import ABC, abstractmethod
from ..models import ProviderResult


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    name: str

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> ProviderResult:
        raise NotImplementedError
