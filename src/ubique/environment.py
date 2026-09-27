from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .state import ROOT, read_json, write_json, utc_now


def _git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    return proc.stdout.strip()


def observe_environment() -> dict[str, Any]:
    """Observe only local/repository facts available to the current runner."""
    source_files = list((ROOT / "src" / "ubique").rglob("*.py"))
    test_files = list((ROOT / "tests").glob("test_*.py"))
    providers = read_json("providers.json", {})
    observation = {
        "timestamp": utc_now(),
        "branch": _git(["branch", "--show-current"]) or "unknown",
        "head": _git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "working_tree_dirty": bool(_git(["status", "--porcelain"])),
        "source_python_files": len(source_files),
        "test_python_files": len(test_files),
        "known_providers": sorted(providers.keys()),
        "repository_exists": (ROOT / ".git").exists(),
    }
    write_json("environment.json", observation)
    return observation
