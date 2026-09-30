from ubique.agent import Agent
from ubique.config import Config
from ubique.models import ProviderResult, Task
import ubique.agent as agent_mod


class FakeGithub:
    def list_tasks(self, max_tasks):
        return []


class FakeRouter:
    def __init__(self):
        self.eligibility_calls = 0

    def remote_eligibility(self):
        self.eligibility_calls += 1
        if self.eligibility_calls == 1:
            return {"groq": {"eligible": True}}
        return {"groq": {"eligible": False}}

    def generate(self, prompt):
        return ProviderResult(
            provider="fallback",
            model="deterministic",
            text="No remote inference provider was available for this cycle.",
        )


def test_midcall_remote_loss_defers_reflection_without_failing(monkeypatch):
    config = Config(
        github_token="",
        github_repository="dabinuss/ubique",
        gemini_api_key="",
        gemini_model="gemini",
        gemini_daily_limit=20,
        groq_api_key="configured",
        groq_model="groq",
        groq_daily_limit=1000,
        hf_token="",
        hf_model="hf",
        hf_endpoint="https://example.invalid",
        max_tasks=3,
        memory_limit=500,
        dry_run=True,
        log_level="INFO",
    )
    agent = Agent(config)
    agent.github = FakeGithub()
    agent.router = FakeRouter()

    state = {}
    episodes = []
    finished = {}

    monkeypatch.setattr(agent_mod, "start_cycle", lambda: {"generation": 42})
    monkeypatch.setattr(agent_mod, "finish_cycle", lambda runtime, summary: finished.update(summary=summary))
    monkeypatch.setattr(agent_mod, "perform_recovery", lambda: {"action_count": 0})
    monkeypatch.setattr(agent_mod, "observe_environment", lambda configured_remote: {})
    monkeypatch.setattr(
        agent_mod,
        "assess_homeostasis",
        lambda memory_limit, configured_remote: {"needs": [], "usable_remote_providers": 1},
    )
    monkeypatch.setattr(agent_mod, "assess_preflight", lambda generation, homeostasis, environment, recovery: {"development_allowed": True})
    monkeypatch.setattr(agent_mod, "cognitive_snapshot", lambda: {"attention": {"next_command": "reflect"}})
    monkeypatch.setattr(agent_mod, "build_curiosity_snapshot", lambda generation: {})
    monkeypatch.setattr(agent_mod, "assess_project_loop", lambda: {"detected": False})
    monkeypatch.setattr(
        agent_mod,
        "autonomous_task",
        lambda *args, **kwargs: Task(
            id="autonomous:reflect:42",
            title="Reflect",
            body="/reflect\ncontinue",
            source="autonomous",
        ),
    )

    def read_json(name, default):
        return state.get(name, default)

    def write_json(name, value):
        state[name] = value

    monkeypatch.setattr(agent_mod, "read_json", read_json)
    monkeypatch.setattr(agent_mod, "write_json", write_json)
    monkeypatch.setattr(agent_mod, "recent_episodes", lambda: [])
    monkeypatch.setattr(agent_mod, "append_episode", lambda episode, limit: episodes.append(episode))
    monkeypatch.setattr(agent_mod, "update_skill", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_mod, "record_self_observation", lambda *args, **kwargs: None)

    def should_not_reset_stagnation(*args, **kwargs):
        raise AssertionError("deferred reflection must not count as completed reflection")

    monkeypatch.setattr(agent_mod, "update_stagnation", should_not_reset_stagnation)

    def defer(generation, reason):
        attention = {
            "generation": generation,
            "focus": "Philosophical self-inquiry",
            "reflection_deferred": True,
            "deferred_reason": reason,
            "next_command": "reflect",
            "next_action": "Retry philosophical reflection when a remote reasoning provider is available.",
        }
        state["attention.json"] = attention
        return attention

    monkeypatch.setattr(agent_mod, "record_reflection_deferred", defer)

    assert agent.run() == 0
    assert finished["summary"] == "ok:handled=1:deferred=1:failed=0"
    assert episodes[-1]["success"] is True
    assert episodes[-1]["deferred"] is True
    assert episodes[-1]["provider"] == "fallback"
    assert state["attention.json"]["reflection_deferred"] is True
    assert state["homeostasis.json"]["usable_remote_providers"] == 0
    assert state["pulse.json"]["should_continue"] is False
    assert "no_usable_remote_reasoning" in state["pulse.json"]["reason"]
