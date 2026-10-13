"""History-derived identity retaining v2 stances and distinguishing experiences."""
from __future__ import annotations
from ..state import read_json
from .storage import Journal

class SelfContinuity:
    def __init__(self,journal:Journal):
        self.journal=journal
        self.old_stances=list(read_json("brain.json",{}).get("current_stances",[]))
        self.history=[]
        for ev in journal.events():
            if ev["kind"]=="self_experience":
                self.history.append(ev["data"])

    def stances(self,new_stances:dict)->list[dict]:
        current={str(s.get("topic")):dict(s,provenance="v2")
                 for s in self.old_stances if s.get("topic")}
        for topic,s in new_stances.items():
            current[topic]=dict(s,provenance="v3")
        return list(current.values())

    def experience(self,kind:str,*,outcome:bool,generation:int,reason:str=""):
        # Behavior is experience; one model completion never rewrites identity.
        record={"action":kind,"success":bool(outcome),"generation":generation,
                "reason":reason[:800],"status":"observed_action_outcome"}
        self.journal.append("self_experience",record,generation=generation)
        self.history.append(record)
        return record

    def skills(self)->dict[str,dict]:
        skills={}
        for record in self.history:
            action=record["action"]
            state=skills.setdefault(action,{"attempts":0,"successes":0})
            state["attempts"]+=1
            state["successes"]+=int(record["success"])
        return skills
