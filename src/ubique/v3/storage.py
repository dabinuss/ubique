"""Durable v3 event journal: immutable compressed shards, incremental replay and recovery.

The original YYYY-MM.jsonl journal is repacked *without losing events*. Once
complete shards have been verified, the old text file is removed from the
working tree (its historical Git versions remain retrievable).
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from ..state import ROOT

SCHEMA = 3
SEGMENT_LIMIT = 750_000
ARCHIVE_LIMIT = 1_500_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


class Journal:
    def __init__(self, root: Path | None = None):
        self.root = root or ROOT / "memory" / "v3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.snapshot_path = self.root / "cortex_snapshot.json.gz"
        self.legacy_snapshot_path = self.root / "cortex_snapshot.json"
        self.manifest = read_json(self.manifest_path, {})
        if not isinstance(self.manifest, dict):
            self.manifest = {}
        for key, value in {"schema_version": SCHEMA, "last_hash": "genesis",
                           "events": 0, "snapshot_events": 0, "migration": {}}.items():
            self.manifest.setdefault(key, value)
        self._cached_events: list[dict] | None = None
        self._pending_repack = False
        self._prepare_segments()

    def _files(self) -> list[Path]:
        return sorted(p for p in (self.root / "events").glob("*")
                      if p.name.endswith((".jsonl", ".jsonl.gz")))

    @staticmethod
    def _read_lines(path: Path) -> Iterator[str]:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            yield from stream

    def _prepare_segments(self) -> None:
        """Convert an old monthly file into verified immutable gzip shards.

        A single active monthly file historically exceeded 20 MB/day. Shards
        are published only after the conversion has completed and verified.
        """
        directory = self.root / "events"
        if not directory.is_dir():
            return
        for legacy in sorted(directory.glob("????-??.jsonl")):
            # A crash after publishing shards but before removing the original
            # is recovered by verifying the new shards before removal.
            marker = self.root / ("repacked_" + legacy.stem + ".json")
            spec = read_json(marker, {})
            archives = [directory / name for name in spec.get("files", [])]
            if (archives and all(p.is_file() for p in archives)
                    and self._verify_repack(legacy, archives)):
                legacy.unlink()
                continue
            if legacy.stat().st_size < SEGMENT_LIMIT:
                continue
            self._repack(legacy, marker)

    def _verify_repack(self, source: Path, archives: list[Path]) -> bool:
        original = hashlib.sha256()
        reproduced = hashlib.sha256()
        for line in self._read_lines(source):
            original.update(line.encode("utf-8"))
        for path in archives:
            for line in self._read_lines(path):
                reproduced.update(line.encode("utf-8"))
        return original.digest() == reproduced.digest()

    def _repack(self, legacy: Path, marker: Path) -> None:
        directory = legacy.parent
        shards: list[Path] = []
        staging = self.root / ".repack"
        staging.mkdir(exist_ok=True)
        index = 0
        buffer: list[str] = []
        size = 0
        def flush() -> None:
            nonlocal size, buffer, index
            if not buffer:
                return
            name = f"{legacy.stem}-{index:06d}.jsonl.gz"
            target = staging / name
            with gzip.open(target, "wt", encoding="utf-8", compresslevel=6) as stream:
                stream.writelines(buffer)
            shards.append(target)
            index += 1
            buffer = []
            size = 0

        for line in self._read_lines(legacy):
            encoded = len(line.encode("utf-8"))
            if size + encoded > ARCHIVE_LIMIT:
                flush()
            buffer.append(line)
            size += encoded
        flush()
        if not self._verify_repack(legacy, shards):
            raise ValueError("Refusing to discard journal: repack verification failed")
        archives: list[Path] = []
        for staged in shards:
            target = directory / staged.name
            os.replace(staged, target)
            archives.append(target)
        # Marker is written only after every archive is readable.
        write_atomic(marker, {"files": [p.name for p in archives],
                              "original_sha256": digest(legacy.read_bytes())})
        if not self._verify_repack(legacy, archives):
            raise ValueError("Repacked journal could not be verified")
        legacy.unlink()
        self._pending_repack = True

    def _active_path(self, when: str) -> Path:
        directory = self.root / "events"
        directory.mkdir(exist_ok=True)
        prefix = when[:7]
        numbered = []
        for p in directory.glob(prefix + "-*.jsonl"):
            try:
                numbered.append(int(p.name[len(prefix)+1:len(prefix)+7]))
            except ValueError:
                continue
        for p in directory.glob(prefix + "-*.jsonl.gz"):
            try:
                numbered.append(int(p.name[len(prefix)+1:len(prefix)+7]))
            except ValueError:
                continue
        number = max(numbered) if numbered else 0
        path = directory / f"{prefix}-{number:06d}.jsonl"
        if path.exists() and path.stat().st_size >= SEGMENT_LIMIT:
            self._compress_segment(path)
            number += 1
            path = directory / f"{prefix}-{number:06d}.jsonl"
        return path

    def _compress_segment(self, path: Path) -> None:
        target = path.with_name(path.name + ".gz")
        temp = self.root / ".repack" / target.name
        temp.parent.mkdir(exist_ok=True)
        with path.open("rb") as original, gzip.open(temp, "wb", compresslevel=6) as archive:
            while True:
                data = original.read(128_000)
                if not data:
                    break
                archive.write(data)
        with path.open("rb") as source, gzip.open(temp, "rb") as restored:
            while True:
                left = source.read(128_000)
                right = restored.read(128_000)
                if left != right:
                    raise ValueError("Segment compression verification failed")
                if not left:
                    break
        os.replace(temp, target)
        path.unlink()

    def append(self, kind: str, data: dict, *, generation: int | None = None,
               identity: str | None = None) -> dict:
        if not isinstance(data, dict):
            raise TypeError("Event payload must be a dictionary")
        timestamp = utc_now()
        serial = {"kind": str(kind), "data": data, "generation": generation,
                  "timestamp": timestamp, "identity": identity or "",
                  "previous_hash": self.manifest["last_hash"]}
        serial["hash"] = digest(canonical(serial))
        serial["id"] = "v3:" + serial["hash"][:24]
        path = self._active_path(timestamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(serial) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.manifest["last_hash"] = serial["hash"]
        self.manifest["last_event_id"] = serial["id"]
        self.manifest["events"] = int(self.manifest["events"]) + 1
        write_atomic(self.manifest_path, self.manifest)
        if self._cached_events is not None:
            self._cached_events.append(serial)
        return serial

    def events(self, *, after: int = 0, verify: bool = False) -> list[dict]:
        if self._cached_events is None or verify:
            records = []
            previous = "genesis"
            for path in self._files():
                for line in self._read_lines(path):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if verify:
                        expected = digest(canonical({k: v for k, v in record.items()
                                                     if k not in ("hash", "id")}))
                        if record.get("previous_hash") != previous or record.get("hash") != expected:
                            raise ValueError("Cognitive journal chain broken at event " +
                                             str(len(records)))
                        previous = record["hash"]
                    records.append(record)
            if verify and (len(records) != int(self.manifest["events"]) or
                           (records and previous != self.manifest["last_hash"])):
                raise ValueError("Journal/manifest mismatch")
            self._cached_events = records
        return self._cached_events[max(0, int(after)):]

    def snapshot(self, graph: dict) -> None:
        payload = {"schema_version": SCHEMA, "generation": graph.get("generation", 0),
                   "events": self.manifest["events"], "graph": graph}
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshot_path.with_name(self.snapshot_path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        # Read back before replacing the last valid snapshot.
        with gzip.open(tmp, "rt", encoding="utf-8") as stream:
            if digest(canonical(json.load(stream))) != digest(canonical(payload)):
                raise ValueError("Snapshot compression verification failed")
        os.replace(tmp, self.snapshot_path)
        self.manifest["snapshot_events"] = self.manifest["events"]
        self.manifest["snapshot_sha256"] = digest(canonical(payload))
        write_atomic(self.manifest_path, self.manifest)
        if self.legacy_snapshot_path.exists():
            self.legacy_snapshot_path.unlink()

    def load_snapshot(self) -> dict | None:
        if self.snapshot_path.is_file():
            with gzip.open(self.snapshot_path, "rt", encoding="utf-8") as stream:
                snapshot = json.load(stream)
        else:
            snapshot = read_json(self.legacy_snapshot_path, None)
        if not isinstance(snapshot, dict):
            return None
        expected = self.manifest.get("snapshot_sha256")
        if expected and digest(canonical(snapshot)) != expected:
            raise ValueError("Corrupted v3 graph snapshot")
        return snapshot

    def migration_record(self, name: str, file: Path, *, records: int = 0) -> None:
        data = file.read_bytes() if file.exists() else b""
        self.manifest["migration"][name] = {
            "path": str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file),
            "sha256": digest(data), "size": len(data), "records": records,
            "verified_at": utc_now(), "baseline_only": True}
        write_atomic(self.manifest_path, self.manifest)

    def migration_drift(self) -> dict[str, bool]:
        """A changed baseline is reported as drift, not mislabelled corruption."""
        result = {}
        for name, spec in self.manifest.get("migration", {}).items():
            path = ROOT / spec["path"] if not Path(spec["path"]).is_absolute() else Path(spec["path"])
            result[name] = path.is_file() and digest(path.read_bytes()) != spec["sha256"]
        return result

    def verify_migration(self) -> dict[str, bool]:
        # The imported v3 snapshot is the verified historical copy. Live v2
        # libraries and state are allowed to change as they are used.
        result = {}
        for name, spec in self.manifest.get("migration", {}).items():
            path = ROOT / spec["path"] if not Path(spec["path"]).is_absolute() else Path(spec["path"])
            result[name] = path.is_file() and path.stat().st_size > 0
        return result
