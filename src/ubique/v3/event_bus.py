"""Typed cognitive event bus, with durable delivery before notifications."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Callable
from .storage import Journal

EVENT_TYPES={"percept","episode","claim","prediction","stance","interest",
             "action_outcome","consolidation","graph_delta","self_experience",
             "change_intent","change_outcome","wish","source_discovered"}

@dataclass
class Event:
    kind:str
    data:dict[str,Any]=field(default_factory=dict)
    generation:int|None=None

class EventBus:
    def __init__(self,journal:Journal):
        self.journal=journal
        self.subscribers:dict[str,list[Callable]]={}

    def subscribe(self,kind:str,handler:Callable):
        if kind not in EVENT_TYPES:raise ValueError("Unknown event")
        self.subscribers.setdefault(kind,[]).append(handler)

    def publish(self,event:Event)->dict:
        if event.kind not in EVENT_TYPES:raise ValueError("Unknown event")
        record=self.journal.append(event.kind,event.data,generation=event.generation)
        # Failure after journaling does not lose the durable event.
        for callback in self.subscribers.get(event.kind,[]):
            callback(record)
        return record
