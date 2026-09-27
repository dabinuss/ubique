from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from .state import ROOT, utc_now

MEMORY_DIR = ROOT / "memory"


def append_episode(episode: dict[str, Any], limit: int = 500) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / "episodes.jsonl"
    episode = {"timestamp": utc_now(), **episode}

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > limit:
        path.write_text("\n".join(lines[-limit:]) + "\n", encoding="utf-8")


def update_skill(command: str, success: bool) -> None:
    path = MEMORY_DIR / "skills.json"
    data = {"skills": {}}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

    skills = data.setdefault("skills", {})
    rec = skills.setdefault(command, {"uses": 0, "successes": 0})
    rec["uses"] = int(rec.get("uses", 0)) + 1
    if success:
        rec["successes"] = int(rec.get("successes", 0)) + 1

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def recent_episodes(limit: int = 8) -> list[dict[str, Any]]:
    path = MEMORY_DIR / "episodes.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def episode_count() -> int:
    """Return the number of retained episode records on disk."""
    path = MEMORY_DIR / "episodes.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def append_memory_record(name: str, record: dict[str, Any], limit: int = 500) -> None:
    """Append one structured JSONL memory record with bounded retention."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / name
    value = {"timestamp": utc_now(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) > limit:
        path.write_text("\n".join(lines[-limit:]) + "\n", encoding="utf-8")


def recent_memory_records(name: str, limit: int = 8) -> list[dict[str, Any]]:
    path = MEMORY_DIR / name
    if not path.exists():
        return []

    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out
