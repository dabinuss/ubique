"""Append-only, hash-linked cognitive events and recoverable graph snapshots.

Git is the durable transport, not an always-on database. Repeated heartbeats
append small events; full graph snapshots are infrequent and replaceable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from ..state import ROOT

SCHEMA = 3

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()

def write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return default

class Journal:
    def __init__(self, root: Path | None = None):
        self.root = root or ROOT / "memory" / "v3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.snapshot_path = self.root / "cortex_snapshot.json"
        self.manifest = read_json(self.manifest_path, {})
        if not isinstance(self.manifest, dict):
            self.manifest = {}
        self.manifest.setdefault("schema_version", SCHEMA)
        self.manifest.setdefault("last_hash", "genesis")
        self.manifest.setdefault("events", 0)
        self.manifest.setdefault("snapshot_events", 0)
        self.manifest.setdefault("migration", {})

    def append(self, kind: str, data: dict, *, generation: int | None = None,
               identity: str | None = None) -> dict:
        if not isinstance(data, dict):
            raise TypeError("Event payload must be a dictionary")
        timestamp = utc_now()
        serial = {"kind": str(kind), "data": data, "generation": generation,
                  "timestamp": timestamp, "identity": identity or ""}
        serial["previous_hash"] = self.manifest["last_hash"]
        serial["hash"] = digest(canonical(serial))
        serial["id"] = "v3:" + serial["hash"][:24]
        segment = timestamp[:7] + ".jsonl"
        path = self.root / "events" / segment
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(serial) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.manifest["last_hash"] = serial["hash"]
        self.manifest["last_event_id"] = serial["id"]
        self.manifest["events"] = int(self.manifest["events"]) + 1
        write_atomic(self.manifest_path, self.manifest)
        return serial

    def events(self, *, after: int = 0, verify: bool = False) -> list[dict]:
        result = []
        previous = "genesis"
        n = 0
        for path in sorted((self.root / "events").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if verify:
                    expected = digest(canonical({
                        k: v for k, v in record.items() if k not in ("hash", "id")
                    }))
                    if record.get("previous_hash") != previous or record.get("hash") != expected:
                        raise ValueError("Cognitive journal chain broken at event " + str(n))
                    previous = record["hash"]
                n += 1
                if n > after:
                    result.append(record)
        if verify and (n != self.manifest.get("events", 0) or
                       (n and previous != self.manifest.get("last_hash"))):
            raise ValueError("Journal/manifest mismatch")
        return result

    def snapshot(self, graph: dict) -> None:
        payload = {"schema_version": SCHEMA, "generation": graph.get("generation", 0),
                   "events": self.manifest["events"], "graph": graph}
        write_atomic(self.snapshot_path, payload)
        self.manifest["snapshot_events"] = self.manifest["events"]
        self.manifest["snapshot_sha256"] = digest(canonical(payload))
        write_atomic(self.manifest_path, self.manifest)

    def load_snapshot(self) -> dict | None:
        snapshot = read_json(self.snapshot_path, None)
        if not isinstance(snapshot, dict):
            return None
        expected = self.manifest.get("snapshot_sha256")
        if expected and digest(canonical(snapshot)) != expected:
            raise ValueError("Corrupted v3 graph snapshot")
        return snapshot

    def migration_record(self, name: str, file: Path, *, records: int = 0) -> None:
        """Reference unmodified v2 evidence without doubling megabytes of history."""
        data = file.read_bytes() if file.exists() else b""
        self.manifest["migration"][name] = {
            "path": str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file),
            "sha256": digest(data), "size": len(data), "records": records,
            "verified_at": utc_now(),
        }
        write_atomic(self.manifest_path, self.manifest)

    def verify_migration(self) -> dict[str, bool]:
        out = {}
        for name, spec in self.manifest.get("migration", {}).items():
            path = Path(spec["path"])
            if not path.is_absolute():
                path = ROOT / path
            out[name] = path.is_file() and digest(path.read_bytes()) == spec["sha256"]
        return out
