import json

from ubique.agent import Agent
from ubique.config import Config
from ubique.models import ProviderResult, Task
import ubique.agent as agent_mod


class FakeGithub:
    def list_tasks(self, max_tasks):
        return []


class SaturationRouter:
    def __init__(self):
        self.generate_calls = 0

    def remote_eligibility(self):
        return {"groq": {"eligible": True}}

    def generate(self, prompt):
        self.generate_calls += 1
        if self.generate_calls == 1:
            payload = {
                "question": "What am I?",
                "interpretation": "I can continue the same line of thought.",
                "assumptions": [],
                "claims": [],
                "provisional_answer": "More reflection may refine the answer.",
                "uncertainty": "High.",
                "next_action": "Reflect again.",
                "next_command": "reflect",
            }
        else:
            payload = {
                "question": "What perspective am I missing?",
                "interpretation": "Further reflection without new material is no longer informative.",
                "assumptions": [],
                "claims": [],
                "provisional_answer": "I need a genuinely different source.",
                "uncertainty": "I do not know which source will change the view.",
                "next_action": "Request a philosophical text on personal identity.",
                "next_command": "library",
                "library_action": "request",
                "library_title": "Reasons and Persons",
                "library_source": "Derek Parfit",
                "library_reason": "Compare a reductionist theory of identity with the recurring current thesis.",
            }
        return ProviderResult(
            provider="groq",
            model="groq",
            text=json.dumps(payload),
        )


def test_saturated_reflection_repairs_reflect_into_source_change(monkeypatch):
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
    agent.router = SaturationRouter()

    state = {}
    episodes = []
    finished = {}
    persisted = {}

    monkeypatch.setattr(agent_mod, "start_cycle", lambda: {"generation": 60})
    monkeypatch.setattr(
        agent_mod,
        "finish_cycle",
        lambda runtime, summary: finished.update(summary=summary),
    )
    monkeypatch.setattr(agent_mod, "perform_recovery", lambda: {"action_count": 0})
    monkeypatch.setattr(agent_mod, "observe_environment", lambda configured_remote: {})
    monkeypatch.setattr(
        agent_mod,
        "assess_homeostasis",
        lambda memory_limit, configured_remote: {"needs": [], "usable_remote_providers": 1},
    )
    monkeypatch.setattr(
        agent_mod,
        "assess_preflight",
        lambda generation, homeostasis, environment, recovery: {"development_allowed": True},
    )
    monkeypatch.setattr(
        agent_mod,
        "cognitive_snapshot",
        lambda: {
            "attention": {"next_command": "reflect"},
            "executed_action_facts": [],
            "reflection_saturation": {
                "detected": True,
                "completed_reflections_since_source_change": 8,
                "budget": 6,
            },
        },
    )
    monkeypatch.setattr(agent_mod, "build_curiosity_snapshot", lambda generation: {})
    monkeypatch.setattr(agent_mod, "assess_project_loop", lambda: {"detected": False})
    monkeypatch.setattr(
        agent_mod,
        "autonomous_task",
        lambda *args, **kwargs: Task(
            id="autonomous:reflect:60",
            title="Break reflection loop",
            body="/reflect\nchoose a new source",
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
    monkeypatch.setattr(
        agent_mod,
        "append_episode",
        lambda episode, limit: episodes.append(episode),
    )
    monkeypatch.setattr(agent_mod, "update_skill", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_mod, "update_stagnation", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_mod, "record_self_observation", lambda *args, **kwargs: None)

    def persist(generation, reflection, observed_facts):
        persisted.update(reflection)
        attention = {
            "generation": generation,
            "focus": "Philosophical self-inquiry",
            "next_command": reflection["next_command"],
            "library_action": reflection.get("library_action", ""),
            "library_title": reflection.get("library_title", ""),
            "library_source": reflection.get("library_source", ""),
            "library_reason": reflection.get("library_reason", ""),
        }
        state["attention.json"] = attention
        return attention

    monkeypatch.setattr(agent_mod, "persist_reflection", persist)

    assert agent.run() == 0
    assert agent.router.generate_calls == 2
    assert persisted["next_command"] == "library"
    assert persisted["library_action"] == "request"
    assert persisted["library_title"] == "Reasons and Persons"
    assert finished["summary"] == "ok:handled=1:deferred=0:failed=0"
    assert episodes[-1]["success"] is True
    assert state["pulse.json"]["should_continue"] is True
    assert state["pulse.json"]["next_command"] == "library"
