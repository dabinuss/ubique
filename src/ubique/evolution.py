from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Any

import httpx

from .state import ROOT


log = logging.getLogger("ubique.evolution")

MAX_FILES = 3
MAX_FILE_BYTES = 30_000

ALLOWED_PREFIXES = (
    "src/ubique/",
    "tests/",
)

DENIED_EXACT = {
    "src/ubique/evolution.py",
    "src/ubique/fzg.py",
    "src/ubique/planner.py",
    "src/ubique/agent.py",
    "src/ubique/github.py",
    "src/ubique/config.py",
    "src/ubique/state.py",
    "src/ubique/providers/router.py",
}

DENIED_PREFIXES = (
    ".github/",
    "state/",
    "memory/",
)

DENIED_NAMES = {
    "pyproject.toml",
    "SECURITY.md",
    "LICENSE",
    ".gitignore",
}


@dataclass(slots=True)
class EvolutionOutcome:
    accepted: bool
    title: str
    summary: str
    branch: str | None = None
    pr_url: str | None = None
    reason: str | None = None
    baseline_score: int | None = None
    candidate_score: int | None = None


def _run(cmd: list[str], timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def _json_from_text(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("proposal did not contain a JSON object")
    return json.loads(value[start:end + 1])


def _safe_path(raw: str) -> Path:
    raw = raw.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or ".." in Path(raw).parts:
        raise ValueError(f"unsafe path: {raw!r}")
    if raw in DENIED_NAMES or raw in DENIED_EXACT:
        raise ValueError(f"protected path: {raw}")
    if any(raw.startswith(prefix) for prefix in DENIED_PREFIXES):
        raise ValueError(f"protected path: {raw}")
    if not any(raw.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        raise ValueError(f"path outside evolution allowlist: {raw}")
    if not raw.endswith(".py"):
        raise ValueError(f"only Python source/tests may evolve: {raw}")
    return ROOT / raw


def validate_proposal(raw_text: str) -> dict[str, Any]:
    proposal = _json_from_text(raw_text)
    title = str(proposal.get("title", "")).strip()
    summary = str(proposal.get("summary", "")).strip()
    changes = proposal.get("changes")

    if not title or len(title) > 120:
        raise ValueError("proposal title is missing or too long")
    if not isinstance(changes, list) or not 1 <= len(changes) <= MAX_FILES:
        raise ValueError(f"proposal must contain 1..{MAX_FILES} changes")

    seen: set[str] = set()
    normalized: list[dict[str, str]] = []

    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("each change must be an object")
        path = str(change.get("path", "")).strip().replace("\\", "/")
        content = change.get("content")
        _safe_path(path)

        if path in seen:
            raise ValueError(f"duplicate path: {path}")
        seen.add(path)

        if not isinstance(content, str):
            raise ValueError(f"content for {path} must be text")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"content too large: {path}")

        lowered = content.lower()
        forbidden_fragments = (
            "os.environ[",
            "subprocess.popen(",
            "eval(",
            "exec(",
            "github_token",
            "gemini_api_key",
            "hf_token",
        )
        if any(fragment in lowered for fragment in forbidden_fragments):
            raise ValueError(f"candidate contains forbidden high-risk fragment: {path}")

        normalized.append({"path": path, "content": content})

    return {"title": title, "summary": summary, "changes": normalized}


def _benchmark() -> int:
    proc = _run(["python", "-m", "ubique.benchmark"], timeout=60)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return int(data["score"])


def _tests_pass() -> tuple[bool, str]:
    commands = [
        ["python", "-m", "compileall", "-q", "src"],
        ["python", "-m", "pytest", "-q"],
    ]
    output: list[str] = []
    for cmd in commands:
        proc = _run(cmd, timeout=180, check=False)
        output.append((proc.stdout + proc.stderr)[-4000:])
        if proc.returncode != 0:
            return False, "\n".join(output)
    return True, "\n".join(output)


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (value or "improvement")[:36]


def _create_draft_pr(token: str, repository: str, title: str, body: str, head: str, base: str) -> str:
    url = f"https://api.github.com/repos/{repository}/pulls"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ubique-autonomous-agent",
    }
    payload = {"title": title, "body": body, "head": head, "base": base, "draft": True}
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"PR creation failed HTTP {r.status_code}: {r.text[:500]}")
    return str(r.json()["html_url"])


def run_evolution(
    proposal_text: str,
    generation: int,
    github_token: str,
    repository: str,
) -> EvolutionOutcome:
    try:
        proposal = validate_proposal(proposal_text)
    except Exception as exc:
        return EvolutionOutcome(False, "Rejected evolution", "", reason=f"proposal rejected: {exc}")

    if not github_token or not repository:
        return EvolutionOutcome(
            False, proposal["title"], proposal["summary"],
            reason="GitHub write context is unavailable"
        )

    original_branch = _run(["git", "branch", "--show-current"]).stdout.strip() or "main"
    baseline_score = _benchmark()
    changed_paths = [c["path"] for c in proposal["changes"]]

    backups: dict[str, bytes | None] = {}
    for change in proposal["changes"]:
        path = _safe_path(change["path"])
        backups[change["path"]] = path.read_bytes() if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(change["content"], encoding="utf-8")

    accepted = False
    branch: str | None = None
    try:
        passed, test_output = _tests_pass()
        if not passed:
            return EvolutionOutcome(
                False, proposal["title"], proposal["summary"],
                reason="candidate tests failed:\n" + test_output[-2500:],
                baseline_score=baseline_score,
            )

        candidate_score = _benchmark()
        if candidate_score < baseline_score:
            return EvolutionOutcome(
                False, proposal["title"], proposal["summary"],
                reason="candidate benchmark regressed",
                baseline_score=baseline_score,
                candidate_score=candidate_score,
            )

        branch = f"ubique/evolve-g{generation}-{_slug(proposal['title'])}"
        existing = _run(["git", "ls-remote", "--heads", "origin", branch], check=False).stdout.strip()
        if existing:
            branch = f"{branch}-{generation}"

        _run(["git", "switch", "-c", branch])
        _run(["git", "add", "--", *changed_paths])
        diff = _run(["git", "diff", "--cached", "--stat"]).stdout.strip()
        if not diff:
            return EvolutionOutcome(
                False, proposal["title"], proposal["summary"],
                reason="proposal produced no source diff",
                baseline_score=baseline_score,
                candidate_score=candidate_score,
            )

        _run(["git", "commit", "-m", f"evolve: {proposal['title']}"])
        _run(["git", "push", "-u", "origin", branch], timeout=120)

        body = (
            "## Autonomous evolution proposal\n\n"
            f"{proposal['summary']}\n\n"
            f"- Generation: `{generation}`\n"
            f"- Baseline benchmark: `{baseline_score}`\n"
            f"- Candidate benchmark: `{candidate_score}`\n"
            "- Compilation: passed\n"
            "- Test suite: passed\n"
            "- Merge mode: **human review required**\n\n"
            "Ubique intentionally creates this as a **draft PR** and never auto-merges "
            "self-generated code."
        )
        pr_url = _create_draft_pr(
            github_token, repository,
            f"[Ubique evolution] {proposal['title']}",
            body, branch, original_branch
        )
        accepted = True
        return EvolutionOutcome(
            True, proposal["title"], proposal["summary"],
            branch=branch, pr_url=pr_url,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
        )

    except Exception as exc:
        return EvolutionOutcome(
            False, proposal["title"], proposal["summary"],
            branch=branch,
            reason=f"evolution pipeline failed safely: {exc}",
            baseline_score=baseline_score,
        )
    finally:
        current = _run(["git", "branch", "--show-current"], check=False).stdout.strip()
        if current != original_branch:
            _run(["git", "switch", original_branch], check=False)

        if not accepted:
            for rel, data in backups.items():
                path = ROOT / rel
                if data is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_bytes(data)
