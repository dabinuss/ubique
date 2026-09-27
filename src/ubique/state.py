from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(name: str, default: Any) -> Any:
    path = STATE_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        # Persistence corruption must not strand the next autonomous cycle.
        # The caller receives its explicit safe default and can reconstruct
        # state from repository/memory evidence.
        return default


def write_json(name: str, value: Any) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def start_cycle() -> dict:
    runtime = read_json("runtime.json", {})
    runtime["generation"] = int(runtime.get("generation", 0)) + 1
    runtime["last_cycle_started_at"] = utc_now()
    runtime["last_result"] = "running"
    write_json("runtime.json", runtime)
    return runtime


def finish_cycle(runtime: dict, result: str) -> None:
    runtime["last_cycle_finished_at"] = utc_now()
    runtime["last_result"] = result
    write_json("runtime.json", runtime)
