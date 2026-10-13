"""Sparse distributed representations and a recurrent, growing synaptic graph."""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable
from collections import defaultdict
from ..state import ROOT
from .storage import read_json

POOL = 8192
ENSEMBLE_SIZE = 40

def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(x)))

def concept_id(label: str) -> str:
    norm = " ".join(label.casefold().split())[:300]
    return "concept:" + hashlib.sha256(norm.encode()).hexdigest()[:20]

def ensemble(label: str, size: int = ENSEMBLE_SIZE) -> tuple[int, ...]:
    """Deterministic sparse neural population; survives process restarts."""
    seed = hashlib.shake_256(label.casefold().strip().encode()).digest(size * 4)
    units = set()
    for p in range(0, len(seed), 4):
        units.add(int.from_bytes(seed[p:p+4], "big") % POOL)
    return tuple(sorted(units))

@dataclass
class Node:
    id: str
    label: str
    kind: str = "concept"
    activation: float = 0.0
    use_count: int = 0
    stability: float = 0.0
    last_active: int = 0
    epistemic_status: str = "memory_index"

@dataclass
class Synapse:
    source: str
    target: str
    relation: str = "associated"
    weight: float = 0.1
    plasticity: float = 0.5
    stability: float = 0.05
    eligibility: float = 0.0
    coactivation_count: int = 0
    last_active: int = 0
    source_event: str = ""
    archived: bool = False

    @property
    def id(self) -> str:
        return hashlib.sha256((self.source+"|"+self.relation+"|"+self.target).encode()).hexdigest()[:24]

class Cortex:
    """Semantic populations + typed associations; all learned edges remain recoverable."""
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Synapse] = {}
        self.tick = 0
        self.changes: list[dict[str, Any]] = []

    @classmethod
    def from_legacy(cls, path=None):
        out = cls()
        raw = read_json(path or ROOT / "memory" / "cortex.json", {})
        for value in raw.get("nodes", []):
            if not isinstance(value, dict) or not value.get("id"):
                continue
            n = Node(id=str(value["id"]), label=str(value.get("label", "")),
                     kind=str(value.get("kind", "concept")),
                     activation=clamp(value.get("activation", 0)),
                     use_count=int(value.get("access_count", 0)),
                     epistemic_status=str(value.get("epistemic_status", "memory_index")))
            out.nodes[n.id] = n
        for value in raw.get("edges", []):
            if not isinstance(value, dict):
                continue
            if value.get("source") not in out.nodes or value.get("target") not in out.nodes:
                continue
            e = Synapse(source=str(value["source"]), target=str(value["target"]),
                relation=str(value.get("relation", "associated")),
                weight=clamp(value.get("weight", 0.2), -1, 1),
                plasticity=clamp(value.get("plasticity", 0.5)),
                stability=clamp(int(value.get("coactivation_count", 0))/30),
                coactivation_count=int(value.get("coactivation_count", 0)))
            out.edges[e.id] = e
        out.changes.clear()
        return out

    @classmethod
    def from_dict(cls, data: dict):
        out=cls()
        out.tick=int(data.get("tick", 0))
        out.nodes={x["id"]:Node(**x) for x in data.get("nodes", [])}
        out.edges={}
        for x in data.get("edges", []):
            e=Synapse(**x);out.edges[e.id]=e
        return out

    def as_dict(self) -> dict:
        return {"generation": self.tick, "tick": self.tick,
                "nodes": [asdict(x) for x in self.nodes.values()],
                "edges": [asdict(x) for x in self.edges.values()]}

    def apply(self, event: dict) -> None:
        kind=event.get("kind")
        data=event.get("data", {})
        if kind=="node_upsert":
            obj=Node(**data);self.nodes[obj.id]=obj
        elif kind=="synapse_upsert":
            obj=Synapse(**data);self.edges[obj.id]=obj
        elif kind=="cortex_tick":
            self.tick=max(self.tick, int(data.get("tick", self.tick)))
        self.changes.clear()

    def _record(self, kind: str, obj) -> None:
        self.changes.append({"kind":kind,"data":asdict(obj)})

    def ensure(self, label: str, *, kind: str="concept", epistemic_status="memory_index") -> Node:
        label=" ".join(str(label).strip().split())[:300]
        if not label:
            raise ValueError("Empty concept")
        cid=concept_id(label)
        node=self.nodes.get(cid)
        if node is None:
            node=Node(cid,label,kind,epistemic_status=epistemic_status)
            self.nodes[cid]=node
            self._record("node_upsert", node)
        return node

    def connect(self, left: str, right: str, *, relation="associated", weight=0.12,
                event_id: str="", bidirectional=False) -> list[Synapse]:
        if left==right or left not in self.nodes or right not in self.nodes:
            return []
        created=[]
        for a,b in [(left,right)]+([(right,left)] if bidirectional else []):
            e=Synapse(a,b,relation,clamp(weight,-1,1),source_event=event_id)
            previous=self.edges.get(e.id)
            if previous:
                previous.archived=False
                e=previous
            else:
                self.edges[e.id]=e
                self._record("synapse_upsert",e)
            created.append(e)
        return created

    def activate(self, labels: Iterable[str], *, strength: float=0.75,
                 kind: str="concept") -> list[str]:
        active=[]
        for label in labels:
            if not str(label).strip():
                continue
            n=self.ensure(str(label),kind=kind)
            n.activation=max(n.activation,clamp(strength))
            n.use_count+=1;n.last_active=self.tick
            self._record("node_upsert",n)
            active.append(n.id)
        return list(dict.fromkeys(active))

    def step(self, *, active_budget: int=120, decay: float=0.82,
             threshold: float=0.035, inhibition: float=0.06) -> list[str]:
        """Local sparse recurrent activation, not a full dense brain simulation."""
        self.tick+=1
        active={n.id:n.activation for n in self.nodes.values() if n.activation>=threshold}
        if not active:
            return []
        incoming=defaultdict(float)
        for edge in self.edges.values():
            if edge.archived or edge.source not in active:
                continue
            incoming[edge.target]+=active[edge.source]*edge.weight*0.36
        for nid,node in self.nodes.items():
            value=decay*node.activation + incoming.get(nid,0)
            # sparse competition/homeostatic inhibition
            value=max(0.0,value-inhibition*max(0.0,len(active)/active_budget-0.5))
            node.activation=clamp(value)
        selected=sorted(self.nodes.values(),key=lambda x:x.activation,reverse=True)[:active_budget]
        kept={node.id for node in selected if node.activation>=threshold}
        for node in self.nodes.values():
            if node.id not in kept:
                node.activation=0.0
            elif node.id in incoming:
                node.last_active=self.tick
            if node.id in active or node.id in incoming:
                self._record("node_upsert",node)
        return list(kept)

    def learn(self, active_ids: Iterable[str], *, reward: float=0.0, rate: float=0.035,
              source_event: str="", pair_budget: int=48) -> dict:
        """Hebbian association, eligibility traces, feedback-gated LTP/LTD."""
        ids=list(dict.fromkeys(n for n in active_ids if n in self.nodes))
        ids.sort(key=lambda n:self.nodes[n].activation,reverse=True)
        ids=ids[:12]
        updates=0;births=0
        for idx,a in enumerate(ids):
            for b in ids[idx+1:]:
                if updates>=pair_budget: break
                new=self.connect(a,b,relation="coactivated",bidirectional=True,event_id=source_event)
                for e in new:
                    if e.coactivation_count==0: births+=1
                    pre=self.nodes[e.source];post=self.nodes[e.target]
                    corr=max(0.03,pre.activation*post.activation)
                    e.eligibility=clamp(0.83*e.eligibility+corr)
                    delta=rate*e.plasticity*(corr + reward*e.eligibility - 0.12*max(0.0,e.weight-0.6))
                    e.weight=clamp(e.weight+delta,-1,1)
                    e.stability=clamp(e.stability+0.005*max(reward,0)+0.001*corr)
                    e.last_active=self.tick;e.coactivation_count+=1
                    self._record("synapse_upsert",e)
                    updates+=1
            if updates>=pair_budget: break
        return {"synapses_created":births,"synapses_updated":updates}

    def feedback(self, active_ids: Iterable[str], outcome: float, *, rate=0.03) -> int:
        ids=set(active_ids);n=0
        for e in self.edges.values():
            if e.eligibility<0.001:
                continue
            if e.source in ids or e.target in ids:
                e.weight=clamp(e.weight+rate*e.plasticity*e.eligibility*clamp(outcome,-1,1),-1,1)
                self._record("synapse_upsert",e);n+=1
            e.eligibility*=0.8
        return n

    def homeostasis(self, *, active_budget: int=120) -> None:
        """Archive unused weak edges, never irreversibly remove them."""
        for e in self.edges.values():
            age=self.tick-e.last_active
            if age>400 and abs(e.weight)<0.10 and e.stability<0.2:
                if not e.archived:
                    e.archived=True;self._record("synapse_upsert",e)
        # no global deletion cap
        candidates=sorted(self.nodes.values(),key=lambda n:n.activation,reverse=True)
        for n in candidates[active_budget:]:
            if n.activation:
                n.activation=0.0;self._record("node_upsert",n)

    def recall(self, cues: Iterable[str], *, limit: int=12) -> list[dict]:
        ids=[]
        for cue in cues:
            token=str(cue).casefold().strip()
            match=next((n for n in self.nodes.values() if n.label.casefold()==token),None)
            if match:ids.append(match.id)
        hits=defaultdict(float)
        for e in self.edges.values():
            if e.source in ids:
                hits[e.target]+=abs(e.weight)*(0.5 if e.archived else 1)
        for node in self.nodes.values():
            if node.id in ids:hits[node.id]+=1.0
        ranked=sorted(hits.items(),key=lambda x:x[1],reverse=True)[:limit]
        return [{"id":i,"label":self.nodes[i].label,"score":round(v,4),
                 "ensemble":ensemble(self.nodes[i].label)[:8]} for i,v in ranked]

    def cross_domain_bridges(self, limit: int=8) -> list[dict]:
        edges=sorted(self.edges.values(),key=lambda x:x.weight,reverse=True)
        bridges=[]
        for e in edges:
            left=self.nodes.get(e.source);right=self.nodes.get(e.target)
            if not left or not right or not left.label or not right.label:
                continue
            if left.label==right.label or left.label in right.label or right.label in left.label:
                continue
            overlap=len(set(ensemble(left.label)) & set(ensemble(right.label)))
            if overlap < 2:
                bridges.append({"left":left.label,"right":right.label,
                                "relation":e.relation,"source_event":e.source_event})
            if len(bridges)>=limit:break
        return bridges
