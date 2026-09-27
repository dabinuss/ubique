from __future__ import annotations

import httpx
from .models import Task


class GitHubClient:
    def __init__(self, token: str, repository: str):
        self.token = token
        self.repository = repository
        self.base = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ubique-autonomous-agent",
        }

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.repository and "/" in self.repository)

    def list_tasks(self, limit: int = 3) -> list[Task]:
        if not self.enabled:
            return []

        url = f"{self.base}/repos/{self.repository}/issues"
        params = {
            "state": "open",
            "labels": "ubique",
            "sort": "created",
            "direction": "asc",
            "per_page": min(max(limit, 1), 20),
        }
        with httpx.Client(timeout=25) as client:
            r = client.get(url, headers=self.headers, params=params)
            r.raise_for_status()
            items = r.json()

        tasks: list[Task] = []
        for item in items:
            if "pull_request" in item:
                continue
            tasks.append(Task(
                id=f"issue:{item['number']}",
                title=item.get("title", ""),
                body=item.get("body") or "",
                number=item["number"],
                author=(item.get("user") or {}).get("login"),
            ))
        return tasks

    def comment(self, issue_number: int, body: str) -> None:
        if not self.enabled:
            return
        url = f"{self.base}/repos/{self.repository}/issues/{issue_number}/comments"
        with httpx.Client(timeout=25) as client:
            r = client.post(url, headers=self.headers, json={"body": body[:60000]})
            r.raise_for_status()

    def remove_label(self, issue_number: int, label: str = "ubique") -> None:
        if not self.enabled:
            return
        url = f"{self.base}/repos/{self.repository}/issues/{issue_number}/labels/{label}"
        with httpx.Client(timeout=25) as client:
            r = client.delete(url, headers=self.headers)
            if r.status_code not in (200, 204, 404):
                r.raise_for_status()
