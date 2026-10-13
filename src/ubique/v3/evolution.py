"""Reversible code-change intentions driven by experience, not performance scores."""
from __future__ import annotations
from ..brain.runtime import NeurocognitiveRuntime
from ..config import Config
from .storage import Journal

class Evolution:
    def __init__(self,journal:Journal,config:Config):
        self.journal=journal;self.config=config

    def attempt(self,reason:str,*,generation:int):
        intent={"reason":reason[:1200],"generation":generation,
                "reversible":True,"provenance":"agent_action"}
        self.journal.append("change_intent",intent,generation=generation)
        success,text,provider=NeurocognitiveRuntime(self.config)._evolve(
            reason,generation)
        result={"intent":intent,"success":success,"provider":provider,
                "result":text[:2500]}
        self.journal.append("change_outcome",result,generation=generation)
        return result
