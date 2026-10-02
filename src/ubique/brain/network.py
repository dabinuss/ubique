from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from ..memory import MEMORY_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalise_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower())[:300]


@dataclass(slots=True)
class SemanticNode:
    id: str
    kind: str
    label: str
    activation: float = 0.0
    excitability: float = 0.5
    threshold: float = 0.18
    access_count: int = 0
    last_activated_at: str | None = None
    epistemic_status: str = "unknown"


@dataclass(slots=True)
class Synapse:
    source: str
    target: str
    relation: str = "associated"
    weight: float = 0.2
    plasticity: float = 0.5
    coactivation_count: int = 0
    last_coactivated_at: str | None = None


class AssociativeNetwork:
    """Persistent semantic graph with spreading activation and Hebbian updates.

    The objects are software representations inspired by neural dynamics.  They
    are not intended to simulate biological cells.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or (MEMORY_DIR / "cortex.json")
        self.nodes: dict[str, SemanticNode] = {}
        self.edges: dict[str, Synapse] = {}
        self.load()

    @staticmethod
    def node_id(kind: str, label: str) -> str:
        normalised = _normalise_label(label)
        digest = hashlib.sha1(f"{kind}:{normalised}".encode("utf-8")).hexdigest()[:16]
        return f"{kind}:{digest}"

    @staticmethod
    def edge_id(source: str, target: str, relation: str) -> str:
        digest = hashlib.sha1(
            f"{source}>{relation}>{target}".encode("utf-8")
        ).hexdigest()[:20]
        return f"syn:{digest}"

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        for value in raw.get("nodes", []):
            if not isinstance(value, dict):
                continue
            try:
                node = SemanticNode(
                    id=str(value["id"]),
                    kind=str(value.get("kind", "concept")),
                    label=str(value.get("label", "")),
                    activation=_clamp(value.get("activation", 0.0)),
                    excitability=_clamp(value.get("excitability", 0.5)),
                    threshold=_clamp(value.get("threshold", 0.18)),
                    access_count=max(0, int(value.get("access_count", 0))),
                    last_activated_at=value.get("last_activated_at"),
                    epistemic_status=str(value.get("epistemic_status", "unknown")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self.nodes[node.id] = node
        for value in raw.get("edges", []):
            if not isinstance(value, dict):
                continue
            try:
                edge = Synapse(
                    source=str(value["source"]),
                    target=str(value["target"]),
                    relation=str(value.get("relation", "associated")),
                    weight=_clamp(value.get("weight", 0.2)),
                    plasticity=_clamp(value.get("plasticity", 0.5)),
                    coactivation_count=max(0, int(value.get("coactivation_count", 0))),
                    last_coactivated_at=value.get("last_coactivated_at"),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self.edges[self.edge_id(edge.source, edge.target, edge.relation)] = edge

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated_at": _utc_now(),
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def ensure_node(
        self,
        label: str,
        *,
        kind: str = "concept",
        excitability: float = 0.5,
        threshold: float = 0.18,
        epistemic_status: str = "unknown",
    ) -> SemanticNode:
        label = _normalise_label(label)
        if not label:
            raise ValueError("semantic node requires a label")
        node_id = self.node_id(kind, label)
        node = self.nodes.get(node_id)
        if node is None:
            node = SemanticNode(
                id=node_id,
                kind=kind[:40] or "concept",
                label=label,
                excitability=_clamp(excitability),
                threshold=_clamp(threshold),
                epistemic_status=epistemic_status[:60] or "unknown",
            )
            self.nodes[node_id] = node
        elif epistemic_status and node.epistemic_status == "unknown":
            node.epistemic_status = epistemic_status[:60]
        return node

    def connect(
        self,
        source: str,
        target: str,
        *,
        relation: str = "associated",
        weight: float = 0.2,
        plasticity: float = 0.5,
        bidirectional: bool = True,
    ) -> None:
        if source == target or source not in self.nodes or target not in self.nodes:
            return

        def upsert(left: str, right: str) -> None:
            key = self.edge_id(left, right, relation)
            edge = self.edges.get(key)
            if edge is None:
                self.edges[key] = Synapse(
                    source=left,
                    target=right,
                    relation=relation[:60] or "associated",
                    weight=_clamp(weight),
                    plasticity=_clamp(plasticity),
                )
            else:
                edge.weight = _clamp(max(edge.weight, weight))
                edge.plasticity = _clamp(max(edge.plasticity, plasticity))

        upsert(source, target)
        if bidirectional:
            upsert(target, source)

    def activate_ids(self, node_ids: Iterable[str], amount: float = 0.7) -> list[str]:
        activated: list[str] = []
        now = _utc_now()
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            effective = _clamp(amount * (0.65 + 0.7 * node.excitability))
            node.activation = min(0.92, max(node.activation, effective))
            node.access_count += 1
            node.last_activated_at = now
            activated.append(node.id)
        return activated

    def activate_labels(
        self,
        labels: Iterable[str],
        *,
        amount: float = 0.7,
        kind: str = "concept",
        epistemic_status: str = "unknown",
    ) -> list[str]:
        ids = [
            self.ensure_node(label, kind=kind, epistemic_status=epistemic_status).id
            for label in labels
            if str(label).strip()
        ]
        return self.activate_ids(ids, amount)

    def spread(
        self,
        *,
        steps: int = 2,
        gain: float = 0.5,
        decay: float = 0.82,
        floor: float = 0.015,
    ) -> None:
        steps = max(0, min(int(steps), 5))
        for _ in range(steps):
            increments: dict[str, float] = {}
            for edge in self.edges.values():
                source = self.nodes.get(edge.source)
                target = self.nodes.get(edge.target)
                if source is None or target is None:
                    continue
                if source.activation < source.threshold:
                    continue
                contribution = source.activation * edge.weight * gain * target.excitability
                if contribution <= floor:
                    continue
                increments[target.id] = increments.get(target.id, 0.0) + contribution

            for node in self.nodes.values():
                node.activation = _clamp(node.activation * decay + increments.get(node.id, 0.0))
                if node.activation < floor:
                    node.activation = 0.0
            self.homeostatic_normalize(target_mean=0.32, ceiling=0.94)

    def decay(self, activation_factor: float = 0.78, edge_factor: float = 0.9995) -> None:
        for node in self.nodes.values():
            node.activation = _clamp(node.activation * activation_factor)
            if node.activation < 0.01:
                node.activation = 0.0
        for edge in self.edges.values():
            if edge.coactivation_count == 0:
                edge.weight = _clamp(edge.weight * edge_factor)

    def hebbian_update(
        self,
        active_ids: Iterable[str],
        learning_rate: float = 0.08,
        max_nodes: int = 6,
    ) -> None:
        ids = [node_id for node_id in dict.fromkeys(active_ids) if node_id in self.nodes]
        ids.sort(key=lambda node_id: self.nodes[node_id].activation, reverse=True)
        ids = ids[: max(2, int(max_nodes))]
        now = _utc_now()
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                self.connect(
                    left,
                    right,
                    relation="coactivated",
                    weight=0.12,
                    plasticity=0.65,
                    bidirectional=True,
                )
                for source, target in ((left, right), (right, left)):
                    key = self.edge_id(source, target, "coactivated")
                    edge = self.edges[key]
                    source_node = self.nodes[source]
                    target_node = self.nodes[target]
                    delta = (
                        learning_rate
                        * edge.plasticity
                        * max(source_node.activation, 0.2)
                        * max(target_node.activation, 0.2)
                    )
                    edge.weight = _clamp(edge.weight + delta)
                    edge.coactivation_count += 1
                    edge.last_coactivated_at = now


    def homeostatic_normalize(
        self,
        *,
        target_mean: float = 0.28,
        ceiling: float = 0.92,
    ) -> None:
        """Divisive normalization prevents a large assembly from saturating."""
        active = [node for node in self.nodes.values() if node.activation > 0.0]
        if not active:
            return
        mean_activation = sum(node.activation for node in active) / len(active)
        scale = min(1.0, max(0.05, target_mean) / max(mean_activation, 1e-9))
        for node in active:
            node.activation = min(ceiling, _clamp(node.activation * scale))
            if node.activation < 0.01:
                node.activation = 0.0


    def neighborhood_labels(
        self,
        labels: Iterable[str],
        *,
        limit: int = 120,
    ) -> list[str]:
        """Return labels in the immediate associative neighborhood of seed labels."""
        wanted = {_normalise_label(label) for label in labels if str(label).strip()}
        seeds = {
            node.id
            for node in self.nodes.values()
            if node.label in wanted
        }
        related: list[str] = []
        seen: set[str] = set()
        for node_id in seeds:
            node = self.nodes.get(node_id)
            if node is not None and node.label not in seen:
                seen.add(node.label)
                related.append(node.label)
        ranked_edges = sorted(
            (
                edge for edge in self.edges.values()
                if edge.source in seeds
            ),
            key=lambda edge: (edge.weight, edge.coactivation_count),
            reverse=True,
        )
        for edge in ranked_edges:
            target = self.nodes.get(edge.target)
            if target is None or target.label in seen:
                continue
            seen.add(target.label)
            related.append(target.label)
            if len(related) >= max(1, int(limit)):
                break
        return related

    def inhibit_labels(
        self,
        labels: Iterable[str],
        *,
        factor: float = 0.15,
        neighbor_factor: float = 0.35,
    ) -> int:
        """Lower activation for a resolved representation and its immediate assembly."""
        wanted = {_normalise_label(label) for label in labels if str(label).strip()}
        seeds = [
            node.id for node in self.nodes.values()
            if node.label in wanted
        ]
        touched: set[str] = set()
        for node_id in seeds:
            node = self.nodes[node_id]
            node.activation = _clamp(node.activation * factor)
            touched.add(node_id)
        for edge in self.edges.values():
            if edge.source not in seeds:
                continue
            target = self.nodes.get(edge.target)
            if target is None:
                continue
            target.activation = _clamp(target.activation * neighbor_factor)
            touched.add(target.id)
        return len(touched)

    def prune(
        self,
        *,
        max_schema_nodes: int = 48,
        max_edges: int = 1200,
        min_weight: float = 0.065,
    ) -> dict[str, int]:
        """Bound graph growth while preserving the strongest recurrent structure."""
        removed_nodes = 0
        schemas = [node for node in self.nodes.values() if node.kind == "schema"]
        schemas.sort(
            key=lambda node: (node.activation, node.access_count, node.last_activated_at or ""),
            reverse=True,
        )
        for node in schemas[max(0, int(max_schema_nodes)):]:
            self.nodes.pop(node.id, None)
            removed_nodes += 1

        valid = set(self.nodes)
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if edge.source in valid and edge.target in valid and edge.weight >= min_weight
        }

        relation_bonus = {"coactivated": 0.04, "pattern_contains": 0.03, "proposed_association": 0.0}
        ranked = sorted(
            self.edges.items(),
            key=lambda item: (
                item[1].weight
                + 0.015 * min(item[1].coactivation_count, 10)
                + relation_bonus.get(item[1].relation, 0.0)
            ),
            reverse=True,
        )
        edge_cap = max(20, int(max_edges))
        removed_edges = max(0, len(ranked) - edge_cap)
        if removed_edges:
            self.edges = dict(ranked[:edge_cap])

        connected: set[str] = set()
        for edge in self.edges.values():
            connected.add(edge.source)
            connected.add(edge.target)
        for node_id, node in list(self.nodes.items()):
            if node.kind == "schema" and node_id not in connected:
                self.nodes.pop(node_id, None)
                removed_nodes += 1

        return {"removed_nodes": removed_nodes, "removed_edges": removed_edges}

    def top_active(self, limit: int = 12, minimum: float = 0.01) -> list[dict[str, Any]]:
        ranked = [
            node for node in self.nodes.values()
            if node.activation >= minimum
        ]
        ranked.sort(
            key=lambda node: (node.activation, node.access_count, node.excitability),
            reverse=True,
        )
        return [
            {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "activation": round(node.activation, 4),
                "epistemic_status": node.epistemic_status,
            }
            for node in ranked[: max(0, limit)]
        ]

    def labels_for_ids(self, node_ids: Iterable[str]) -> list[str]:
        return [
            self.nodes[node_id].label
            for node_id in node_ids
            if node_id in self.nodes
        ]

    def strongest_edges(self, limit: int = 20) -> list[dict[str, Any]]:
        ranked = sorted(
            self.edges.values(),
            key=lambda edge: (edge.weight, edge.coactivation_count),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for edge in ranked[: max(0, limit)]:
            source = self.nodes.get(edge.source)
            target = self.nodes.get(edge.target)
            if source is None or target is None:
                continue
            out.append({
                "source": source.label,
                "target": target.label,
                "relation": edge.relation,
                "weight": round(edge.weight, 4),
                "coactivations": edge.coactivation_count,
            })
        return out
