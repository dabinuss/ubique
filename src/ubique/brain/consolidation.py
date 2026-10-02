from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..memory import MEMORY_DIR
from .episodic import EpisodeStore
from .network import AssociativeNetwork

if TYPE_CHECKING:
    from .substrate import CognitiveSubstrateManager


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Consolidator:
    def __init__(
        self,
        episodes: EpisodeStore,
        network: AssociativeNetwork,
        substrate: "CognitiveSubstrateManager | None" = None,
        simulations_path: Path | None = None,
    ):
        self.episodes = episodes
        self.network = network
        self.substrate = substrate
        self.simulations_path = simulations_path or (MEMORY_DIR / "simulations.jsonl")

    def nrem(self, *, learning_rate: float = 0.08) -> dict[str, Any]:
        replay = self.episodes.replay_candidates(10)
        pair_counts: Counter[tuple[str, str]] = Counter()
        replayed_ids: list[str] = []

        for episode in replay:
            concepts = [
                str(value).strip().lower()
                for value in episode.get("concepts", [])
                if str(value).strip()
                and len(str(value).strip()) <= 80
                and len(str(value).strip().split()) <= 8
                and not str(value).strip().endswith("?")
            ][:8]
            if not concepts:
                continue
            node_ids = self.network.activate_labels(
                concepts,
                amount=0.42 + 0.35 * float(episode.get("salience", 0.0)),
                kind="concept",
                epistemic_status="memory_index",
            )
            self.network.hebbian_update(
                node_ids,
                learning_rate=min(learning_rate, 0.05),
                max_nodes=5,
            )
            replayed_ids.append(str(episode.get("id", "")))
            for left, right in itertools.combinations(sorted(set(concepts)), 2):
                pair_counts[(left, right)] += 1

        schema_nodes = 0
        schema_candidates = [
            (pair, count)
            for pair, count in pair_counts.items()
            if count >= 3
        ]
        schema_candidates.sort(key=lambda item: item[1], reverse=True)
        for (left, right), count in schema_candidates[:8]:
            schema = self.network.ensure_node(
                f"{left} ↔ {right}",
                kind="schema",
                excitability=0.45,
                epistemic_status="consolidated_pattern",
            )
            left_node = self.network.ensure_node(left, kind="concept", epistemic_status="memory_index")
            right_node = self.network.ensure_node(right, kind="concept", epistemic_status="memory_index")
            strength = min(0.75, 0.18 + 0.09 * count)
            self.network.connect(schema.id, left_node.id, relation="pattern_contains", weight=strength)
            self.network.connect(schema.id, right_node.id, relation="pattern_contains", weight=strength)
            schema_nodes += 1

        self.network.decay(activation_factor=0.62, edge_factor=0.996)
        self.network.homeostatic_normalize(target_mean=0.24, ceiling=0.88)
        pruning = self.network.prune(max_schema_nodes=48, max_edges=1200)
        self.network.save()
        return {
            "mode": "nrem",
            "replayed_episode_ids": replayed_ids,
            "schema_nodes_reinforced": schema_nodes,
            "pruning": pruning,
        }

    def _append_simulation(self, record: dict[str, Any]) -> dict[str, Any]:
        self.simulations_path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "timestamp": utc_now(),
            "epistemic_status": "imagined",
            **record,
        }
        with self.simulations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        lines = self.simulations_path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 500:
            self.simulations_path.write_text("\n".join(lines[-500:]) + "\n", encoding="utf-8")
        return value

    def rem(self) -> dict[str, Any]:
        active = self.network.top_active(limit=10, minimum=0.03)
        labels = [str(item.get("label", "")) for item in active if str(item.get("label", "")).strip()]
        if len(labels) < 2:
            recent = self.episodes.replay_candidates(6)
            labels = list(dict.fromkeys(
                str(concept)
                for episode in recent
                for concept in episode.get("concepts", [])
                if str(concept).strip()
            ))[:10]

        if len(labels) < 2:
            return {"mode": "rem", "created": 0, "reason": "insufficient_active_material"}

        left, right = labels[0], labels[-1]
        imagined_text = (
            f"Counterfactual association to explore later: what changes if '{left}' "
            f"and '{right}' are related more strongly than current evidence establishes?"
        )
        source = "deterministic_recombination"

        if self.substrate is not None and self.substrate.eligible_names():
            prompt = f"""Generate exactly one explicitly imaginary counterfactual connecting two active concepts.
Do not state it as fact or observation. Do not recommend an action. Return strict JSON using the normal cognitive packet schema.
Active concepts: {json.dumps([left, right], ensure_ascii=False)}
Put the imagined proposition in hypotheses and set uncertainty high."""
            packets = self.substrate.sample(prompt)
            if packets:
                hypotheses = packets[0].get("hypotheses", [])
                questions = packets[0].get("questions", [])
                candidate = (hypotheses or questions or [""])[0]
                if str(candidate).strip():
                    imagined_text = str(candidate).strip()[:3000]
                    source = f"substrate:{packets[0].get('provider', 'unknown')}"

        simulation = self._append_simulation({
            "kind": "counterfactual_recombination",
            "source": source,
            "concepts": [left, right],
            "text": imagined_text,
        })
        self.episodes.append(
            kind="simulation",
            text=imagined_text,
            source=source,
            concepts=[left, right],
            epistemic_status="imagined",
            novelty=0.7,
            surprise=0.5,
            salience=0.3,
            payload={"simulation_timestamp": simulation["timestamp"]},
        )
        self.network.decay(activation_factor=0.58, edge_factor=0.999)
        self.network.save()
        return {
            "mode": "rem",
            "created": 1,
            "source": source,
            "concepts": [left, right],
        }

    def run(self, mode: str) -> dict[str, Any]:
        return self.rem() if str(mode).lower() == "rem" else self.nrem()
