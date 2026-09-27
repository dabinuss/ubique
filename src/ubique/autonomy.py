from __future__ import annotations

from .models import Task


def autonomous_task(generation: int, remote_reasoning_available: bool) -> Task:
    """Create Ubique's endogenous task for this generation.

    The heartbeat may run frequently, but expensive/remote reasoning is
    deliberately sparse so that Ubique can remain active on free tiers.
    """

    # Every 24th generation (~6h at a 15-minute heartbeat) attempt one bounded
    # evolution. This is the highest-cost endogenous cycle.
    if remote_reasoning_available and generation % 24 == 0:
        body = """/evolve
You are in an endogenous Ubique evolution cycle.

System S:
Ubique as the GitHub-native agent consisting of its source code, persistent
state/memory, provider router, GitHub Actions heartbeat and bounded evolution
pipeline.

Normative basis:
FZG v1.0. Do not revise the FZG theory. Preserve the separation of
self-preservation P, functional goal-directedness G_A and the non-scalar
intelligence profile I_C=(Z,K,R,L).

A_self:
Maintain operational continuity across independent GitHub Actions runs while
preserving valid state and the ability to execute future cycles.

Development objective for this cycle:
Use recent episodic memory and the current implementation to identify ONE
small, testable improvement that increases reliable functional capability,
adaptation, observability, resource efficiency, or solution-path diversity
without weakening the protected FZG core or safety/evolution gates.

Return only the strict evolution JSON required by the evolution pipeline.
Do not change FZG definitions merely to make Ubique appear more capable.
"""
        return Task(
            id=f"autonomous:evolve:{generation}",
            title=f"Endogenous evolution generation {generation}",
            body=body,
            source="autonomous",
        )

    # Every 6th generation (~90 min) perform a remote FZG self-analysis.
    if remote_reasoning_available and generation % 6 == 0:
        body = """/fzg
Perform an endogenous FZG v1.0 self-analysis of Ubique.

S:
Ubique as the GitHub-native agent consisting of source code, persistent
state/memory, provider routing, heartbeat execution and bounded evolution.

A:
Reliable completion of one autonomous cycle while preserving the capacity for
future cycles. Treat operational continuity as A_self only where appropriate;
do not infer intelligence from self-preservation.

C:
The current public GitHub repository and a GitHub-hosted Actions execution
cycle with the providers and memory actually available in this generation.

Q:
Use only evidence available from current state, provider ledger, recent
episodes, tests and observable repository behavior. Identify what would count
as success and what remains unmeasured.

M_S:
The planner-provider-executor-memory loop plus the heartbeat/evolution
mechanisms actually active in the repository.

M_S^-:
Define a meaningful control or ablation for the specific claim being examined,
without arbitrarily destroying the entire system.

Determine the causal-identification status of G_A honestly. Then, and only
then, characterize Z, K, R and L separately. Identify one concrete empirical
weakness or missing measurement that a future generation could improve.
Do not invent a global intelligence score.
"""
        return Task(
            id=f"autonomous:fzg:{generation}",
            title=f"Endogenous FZG self-analysis generation {generation}",
            body=body,
            source="autonomous",
        )

    # Cheap heartbeat cycle: keep continuity and observability without an LLM.
    return Task(
        id=f"autonomous:status:{generation}",
        title=f"Autonomous heartbeat generation {generation}",
        body="/status",
        source="autonomous",
    )
