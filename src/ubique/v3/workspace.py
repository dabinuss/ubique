"""Finite global workspace assembling active distributed concepts."""
from __future__ import annotations
from dataclasses import dataclass
from .representation import Cortex
from .evidence import is_system_topic

@dataclass
class WorkspaceItem:
    node_id:str
    label:str
    activation:float
    novelty:float
    conflict:float
    score:float

class Workspace:
    def __init__(self,capacity=7):
        self.capacity=max(3,min(12,int(capacity)))
        self.items:list[WorkspaceItem]=[]

    def compete(self,cortex:Cortex,*,interest=None)->list[WorkspaceItem]:
        interest=interest or {}
        ranked=[]
        for n in cortex.nodes.values():
            if n.activation<.03:continue
            age=max(0,cortex.tick-n.last_active)
            novelty=1.0/(1.0+n.use_count*.045)
            salience=float(interest.get(n.label,0))
            score=.52*n.activation+.27*novelty+.21*salience-.02*min(5,age)
            if n.id.startswith(("schema:","issue:")) or is_system_topic(n.label):
                # Preserve autobiographical technical memories but let topics
                # of life, science, history etc. compete for attention.
                score*=.12 if salience<.6 else .65
            ranked.append(WorkspaceItem(n.id,n.label,n.activation,novelty,0,score))
        ranked.sort(key=lambda x:x.score,reverse=True)
        self.items=ranked[:self.capacity]
        return self.items

    def focus_labels(self)->list[str]:
        return [item.label for item in self.items]

    def snapshot(self)->list[dict]:
        return [vars(item).copy() for item in self.items]
