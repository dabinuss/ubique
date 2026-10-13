"""Interleaved NREM/REM-inspired learning without external providers."""
from __future__ import annotations
from .hippocampus import Hippocampus,terms
from .representation import Cortex
from .world_model import WorldModel
from .storage import Journal

class Consolidator:
    def __init__(self,journal:Journal,hippocampus:Hippocampus,
                 cortex:Cortex,world:WorldModel):
        self.journal=journal;self.hippocampus=hippocampus
        self.cortex=cortex;self.world=world

    def nrem(self,*,generation:int,limit=8)->dict:
        replay=self.hippocampus.replay(limit,generation)
        seen=set();pairs=0
        for ep in replay:
            concepts=list(ep.get("concepts") or terms(ep.get("text","")))[:7]
            active=self.cortex.activate(concepts,strength=.55)
            result=self.cortex.learn(active,rate=.018,source_event=ep.get("id",""))
            pairs+=result["synapses_updated"]
            seen.add(ep.get("id",""))
        self.cortex.homeostasis()
        result={"mode":"nrem","replayed":len(seen),"synapse_updates":pairs,
                "source_episode_ids":list(seen)}
        self.journal.append("consolidation",result,generation=generation)
        return result

    def rem(self,*,generation:int)->dict:
        """Analogical paths are simulated hypotheses, not new observations."""
        bridges=self.cortex.cross_domain_bridges(5)
        records=[]
        for bridge in bridges[:2]:
            statement=("Possible relation between "+bridge["left"]+" and "+
                       bridge["right"]+" (requires independent evidence).")
            claim=self.world.claim(statement,status="SIMULATED",
                source_refs=[bridge["source_event"]] if bridge["source_event"] else [],
                confidence=0.12,generation=generation)
            records.append(claim["id"])
        result={"mode":"rem","simulations":records}
        self.journal.append("consolidation",result,generation=generation)
        return result
