import json
import os
from pathlib import Path

class Ledger:
    def __init__(self, path: str = "ledger.json"):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {"successes": 0, "failures": 0, "entries": []}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return {"successes": 0, "failures": 0, "entries": []}

    def save(self, data: dict) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"Warning: Failed to persist ledger: {e}")
