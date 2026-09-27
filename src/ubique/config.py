from __future__ import annotations

from dataclasses import dataclass
import os


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Config:
    github_token: str
    github_repository: str
    gemini_api_key: str
    gemini_model: str
    hf_token: str
    hf_model: str
    hf_endpoint: str
    max_tasks: int
    memory_limit: int
    dry_run: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            github_token=os.getenv("GITHUB_TOKEN", ""),
            github_repository=os.getenv("GITHUB_REPOSITORY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            hf_endpoint=os.getenv(
                "HF_ENDPOINT",
                "https://router.huggingface.co/hf-inference/models",
            ).rstrip("/"),
            max_tasks=max(1, min(int(os.getenv("UBIQUE_MAX_TASKS", "3")), 20)),
            memory_limit=max(10, int(os.getenv("UBIQUE_MEMORY_LIMIT", "500"))),
            dry_run=_truthy(os.getenv("UBIQUE_DRY_RUN")),
            log_level=os.getenv("UBIQUE_LOG_LEVEL", "INFO").upper(),
        )
