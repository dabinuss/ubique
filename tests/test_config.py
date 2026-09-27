from ubique.config import Config


def test_defaults(monkeypatch):
    for key in [
        "GITHUB_TOKEN", "GITHUB_REPOSITORY", "GEMINI_API_KEY", "HF_TOKEN",
        "UBIQUE_MAX_TASKS", "UBIQUE_MEMORY_LIMIT", "UBIQUE_DRY_RUN"
    ]:
        monkeypatch.delenv(key, raising=False)
    c = Config.from_env()
    assert c.max_tasks == 3
    assert c.memory_limit == 500
    assert c.dry_run is False


def test_task_limit_is_bounded(monkeypatch):
    monkeypatch.setenv("UBIQUE_MAX_TASKS", "999")
    assert Config.from_env().max_tasks == 20
