from __future__ import annotations

import json
import logging

from .config import Config
from .autonomy import autonomous_task
from .github import GitHubClient
from .homeostasis import assess_homeostasis
from .environment import observe_environment
from .fzg_telemetry import measure_fzg_telemetry
from .recovery import perform_recovery
from .memory import append_episode, recent_episodes, update_skill
from .planner import make_prompt, parse_task
from .evolution import run_evolution
from .state import finish_cycle, read_json, start_cycle, write_json, utc_now
from .providers.fallback import FallbackProvider
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .providers.huggingface import HuggingFaceProvider
from .providers.router import ProviderRouter


log = logging.getLogger("ubique")


def make_evolution_repair_prompt(original_proposal: str, failure_reason: str) -> str:
    return f"""Your previous autonomous evolution proposal failed validation or tests.

Failure:
{failure_reason[:5000]}

Original proposal:
{original_proposal[:14000]}

Return ONE corrected proposal as strict JSON only, using the same schema:
{{"title":"...","summary":"...","changes":[{{"path":"src/ubique/<file>.py","content":"complete file contents"}}]}}

Rules:
- Fix the concrete error instead of redesigning unrelated parts.
- Keep the change minimal.
- Never modify protected FZG policy, autonomy/evolution gates, workflows, secrets,
  dependency metadata, state or memory.
- Output JSON only; no markdown fences or explanation.
"""


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.github = GitHubClient(config.github_token, config.github_repository)
        self.router = ProviderRouter(
            [
                GeminiProvider(config.gemini_api_key, config.gemini_model),
                GroqProvider(config.groq_api_key, config.groq_model),
                HuggingFaceProvider(config.hf_token, config.hf_model, config.hf_endpoint),
                FallbackProvider(),
            ],
            daily_limits={
                "gemini": config.gemini_daily_limit,
                "groq": config.groq_daily_limit,
            },
        )

    def status_text(self, generation: int) -> str:
        providers = read_json("providers.json", {})
        homeostasis = read_json("homeostasis.json", {})
        telemetry = read_json("fzg_telemetry.json", {})
        environment = read_json("environment.json", {})
        return (
            f"Ubique generation: {generation}\n\n"
            "Provider ledger:\n"
            f"```json\n{json.dumps(providers, indent=2)}\n```\n\n"
            "Homeostasis:\n"
            f"```json\n{json.dumps(homeostasis, indent=2)[:3000]}\n```\n\n"
            "FZG telemetry:\n"
            f"```json\n{json.dumps(telemetry, indent=2)[:3000]}\n```\n\n"
            "Environment:\n"
            f"```json\n{json.dumps(environment, indent=2)[:2000]}\n```"
        )

    def run(self) -> int:
        runtime = start_cycle()
        generation = int(runtime["generation"])
        log.info("Starting Ubique generation %s", generation)

        # Deterministic self-maintenance and measurement happen before goal
        # selection so autonomous behavior is grounded in current evidence.
        recovery = perform_recovery()
        configured_remote = [
            name
            for name, configured in (
                ("gemini", bool(self.config.gemini_api_key)),
                ("groq", bool(self.config.groq_api_key)),
                ("huggingface", bool(self.config.hf_token)),
            )
            if configured
        ]
        environment = observe_environment(configured_remote=configured_remote)
        homeostasis = assess_homeostasis(
            self.config.memory_limit,
            configured_remote=configured_remote,
        )
        telemetry = measure_fzg_telemetry()
        log.info(
            "Self-state measured: recovery=%s usable_remote=%s",
            recovery.get("action_count", 0),
            homeostasis.get("usable_remote_providers", 0),
        )

        handled = 0
        failed = 0

        try:
            tasks = self.github.list_tasks(self.config.max_tasks)
            remote_reasoning_available = bool(
                self.config.gemini_api_key
                or self.config.groq_api_key
                or self.config.hf_token
            )
            endogenous = autonomous_task(
                generation,
                remote_reasoning_available=remote_reasoning_available,
                homeostasis=homeostasis,
                telemetry=telemetry,
                environment=environment,
            )
            tasks.append(endogenous)
            write_json("current_goal.json", {
                "timestamp": utc_now(),
                "generation": generation,
                "task_id": endogenous.id,
                "title": endogenous.title,
                "command": parse_task(endogenous).command,
                "source": endogenous.source,
            })
            log.info(
                "Discovered %s task(s), including endogenous autonomous cycle",
                len(tasks),
            )

            for task in tasks:
                planned = parse_task(task)
                provider_name = "deterministic"

                try:
                    if planned.command == "status":
                        result_text = self.status_text(generation)
                    else:
                        prompt = make_prompt(planned.command, planned.payload, recent_episodes())
                        result = self.router.generate(prompt)
                        result_text = result.text
                        provider_name = result.provider

                        if planned.command == "evolve":
                            if provider_name == "fallback":
                                raise RuntimeError(
                                    "self-evolution requires a remote reasoning provider; "
                                    "the deterministic fallback cannot author code"
                                )
                            original_proposal = result_text
                            evo = run_evolution(
                                original_proposal,
                                generation,
                                self.config.github_token,
                                self.config.github_repository,
                            )

                            repaired = False
                            if (
                                not evo.accepted
                                and evo.reason
                                and (
                                    evo.reason.startswith("candidate tests failed:")
                                    or evo.reason.startswith("proposal rejected:")
                                )
                            ):
                                repair_prompt = make_evolution_repair_prompt(
                                    original_proposal,
                                    evo.reason,
                                )
                                repair_result = self.router.generate(repair_prompt)
                                if repair_result.provider != "fallback":
                                    provider_name = repair_result.provider
                                    repaired = True
                                    evo = run_evolution(
                                        repair_result.text,
                                        generation,
                                        self.config.github_token,
                                        self.config.github_repository,
                                    )

                            if not evo.accepted:
                                raise RuntimeError(evo.reason or "evolution proposal rejected")

                            result_text = (
                                f"Evolution candidate validated and published as draft PR.\n\n"
                                f"- Title: {evo.title}\n"
                                f"- Branch: `{evo.branch}`\n"
                                f"- Baseline benchmark: `{evo.baseline_score}`\n"
                                f"- Candidate benchmark: `{evo.candidate_score}`\n"
                                f"- Self-repair used: `{repaired}`\n"
                                f"- Pull request: {evo.pr_url}\n\n"
                                "Human review is required before merge."
                            )

                    if not self.config.dry_run and task.number is not None:
                        reply = (
                            f"### Ubique generation {generation}\n\n"
                            f"Provider: `{provider_name}`\n\n"
                            f"{result_text}\n\n"
                            "---\n"
                            "_Processed autonomously by GitHub Actions._"
                        )
                        self.github.comment(task.number, reply)
                        self.github.remove_label(task.number, "ubique")

                    append_episode({
                        "generation": generation,
                        "task_id": task.id,
                        "command": planned.command,
                        "provider": provider_name,
                        "success": True,
                        "result": result_text[:1200],
                    }, self.config.memory_limit)
                    update_skill(planned.command, True)
                    handled += 1

                except Exception as exc:
                    failed += 1
                    log.exception("Task %s failed", task.id)
                    append_episode({
                        "generation": generation,
                        "task_id": task.id,
                        "command": planned.command,
                        "provider": provider_name,
                        "success": False,
                        "result": str(exc)[:800],
                    }, self.config.memory_limit)
                    update_skill(planned.command, False)

                    if not self.config.dry_run and task.number is not None:
                        try:
                            self.github.comment(
                                task.number,
                                f"### Ubique generation {generation}\n\n"
                                "Task failed safely and remains labelled for retry.\n\n"
                                f"`{type(exc).__name__}: {str(exc)[:500]}`"
                            )
                        except Exception:
                            log.exception("Could not report task failure")

            summary = f"ok:handled={handled}:failed={failed}"
            finish_cycle(runtime, summary)
            log.info(summary)
            return 0 if failed == 0 else 1

        except Exception as exc:
            finish_cycle(runtime, f"cycle_failed:{type(exc).__name__}")
            log.exception("Cycle failed")
            return 2
