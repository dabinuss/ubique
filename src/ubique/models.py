from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Task:
    id: str
    title: str
    body: str
    source: str = "github"
    number: int | None = None
    author: str | None = None


@dataclass(slots=True)
class ProviderResult:
    provider: str
    text: str
    model: str | None = None
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class PlannedTask:
    command: str
    payload: str
