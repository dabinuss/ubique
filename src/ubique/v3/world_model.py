"""Epistemic graph: keep source claims, hypotheses, predictions and stances distinct."""
from __future__ import annotations
import hashlib
from typing import Any
from .storage import Journal

def key(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()[:20]

STATUSES={"OBSERVED","SOURCE_CLAIM","INFERRED","HYPOTHESIS","SIMULATED",
          "DISPUTED","CORROBORATED","REJECTED","STALE"}

class WorldModel:
    def __init__(self,journal:Journal):
        self.journal=journal
        self.claims={}
        self.predictions={}
        self.stances={}
        for ev in journal.events():
            d=ev.get("data",{})
            if ev["kind"]=="claim":self.claims[d["id"]]=d
            if ev["kind"]=="prediction":self.predictions[d["id"]]=d
            if ev["kind"]=="stance":self.stances[d["topic"]]=d

    def claim(self,text:str,*,status="HYPOTHESIS",source_refs=(),
              confidence=0.4,counter_refs=(),generation=None) -> dict:
        if status not in STATUSES:raise ValueError("Unknown epistemic status")
        if not text.strip():raise ValueError("Empty claim")
        cid="claim:"+key(text)
        former=self.claims.get(cid,{})
        sources=list(dict.fromkeys(list(former.get("source_refs",[]))+list(source_refs)))
        out={"id":cid,"statement":text[:2800],"status":status,
             "source_refs":sources[:32],"counter_refs":list(counter_refs)[:32],
             "confidence":min(.98,max(.01,float(confidence))),
             "revisions":int(former.get("revisions",0))+1}
        # Absence of independent evidence must not promote an hypothesis.
        if former and not set(source_refs)-set(former.get("source_refs",[])):
            out["confidence"]=former.get("confidence",out["confidence"])
        self.journal.append("claim",out,generation=generation)
        self.claims[cid]=out
        return out

    def prediction(self,statement:str,*,expected:str,context:str="",generation=None):
        id_="prediction:"+key(statement+expected)
        record={"id":id_,"statement":statement[:1500],"expected":expected[:1500],
                "context":context[:2000],"outcome":None,"status":"pending"}
        self.journal.append("prediction",record,generation=generation)
        self.predictions[id_]=record
        return record

    def resolve_prediction(self,identifier:str,*,observed:str,match:bool,generation=None):
        if identifier not in self.predictions:raise KeyError(identifier)
        pred=dict(self.predictions[identifier])
        pred.update({"outcome":observed[:1800],"status":"confirmed" if match else "refuted"})
        self.journal.append("prediction",pred,generation=generation)
        self.predictions[identifier]=pred
        return pred

    def stance(self,topic:str,position:str,*,evidence_ids=(),relation="new",
               uncertainty=0.5,generation=None):
        if not list(evidence_ids):raise ValueError("Stances require provenance")
        old=self.stances.get(topic,{})
        record={"topic":topic,"position":position[:3000],"evidence_ids":list(evidence_ids)[:20],
                "relation":relation,"uncertainty":max(0,min(1,float(uncertainty))),
                "previous_position":old.get("position"),"revisable":True}
        self.journal.append("stance",record,generation=generation)
        self.stances[topic]=record
        return record

    def contradictions(self) -> list[tuple[dict,dict]]:
        # No unsupported assertion of logical contradiction from textual similarity.
        # Explicitly disputed claims can be retrieved for deliberation.
        return [(x,y) for x in self.claims.values() for y in self.claims.values()
                if x["id"]!=y["id"] and
                x["id"] in y.get("counter_refs",[])][:30]
