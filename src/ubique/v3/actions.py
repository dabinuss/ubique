"""Typed autonomous actions; unlike v2 the supported library sync action is shared."""
from __future__ import annotations
from dataclasses import dataclass,field
import json
from typing import Any,Callable
from ..library import apply_library_action,library_catalog
from ..experiments import run_experiment, SAFE_EXPERIMENTS
from .hippocampus import terms
from .representation import Cortex
from .curiosity import Curiosity
from .storage import Journal
from .consolidation import Consolidator

LIBRARY_ACTIONS={"list","read","add","request","note","sync"}
ACTION_KINDS={"read","search","explore","connect","question","experiment","simulate",
              "note","revise_stance","practice_skill","project","edit_code",
              "communicate","rest","consolidate","attend","library","evolve"}
@dataclass
class Candidate:
    kind:str
    reason:str
    payload:dict[str,Any]=field(default_factory=dict)
    score:float=.5
    source:str="local"

@dataclass
class Result:
    kind:str
    success:bool
    effect:str
    evidence_ids:list[str]=field(default_factory=list)
    provider:str="local"
    payload:dict[str,Any]=field(default_factory=dict)

class Registry:
    def __init__(self):
        self.handlers={}
    def register(self,name:str,handler:Callable):
        if name not in ACTION_KINDS:raise ValueError("Unregistered action kind")
        self.handlers[name]=handler
    def execute(self,candidate:Candidate,ctx:dict)->Result:
        kind=candidate.kind
        if kind not in ACTION_KINDS or kind not in self.handlers:
            return Result(kind,False,"Action has no executor")
        try:return self.handlers[kind](candidate.payload,ctx)
        except Exception as exc:return Result(kind,False,type(exc).__name__+": "+str(exc)[:500])

def reading_candidates(curiosity:Curiosity)->list[Candidate]:
    catalog=library_catalog(200)
    available=[x for x in catalog if x.get("kind")=="book"
               and not x.get("fully_read",False)
               and (x.get("content_available") or x.get("content_path") or x.get("source_url"))]
    result=[]
    for x in available:
        topic=str(x.get("title",""))
        interest=curiosity.score(topic)
        repeats=int(x.get("read_count",0))
        # A book with an established reading history remains attractive so
        # reading progresses beyond opening chapters, without monopolising focus.
        depth=.16 if 2<=repeats<=90 else (.09 if repeats==1 else 0)
        result.append(Candidate("read","Continue actual source text: "+topic,
           {"item_id":x.get("id"),"title":topic},
           score=.60+interest*.25+depth-.004*min(90,repeats),source="library"))
    return sorted(result,key=lambda x:x.score,reverse=True)[:6]

def build_registry()->Registry:
    reg=Registry()
    def read(spec,ctx):
        if not spec.get("item_id"):return Result("read",False,"missing item_id")
        record=apply_library_action({"action":"read","item_id":spec["item_id"],
                                     "max_chars":8000},actor="ubique")
        if not record.get("content_available") or not record.get("excerpt"):
            return Result("read",False,"Source unavailable or already complete")
        excerpt=record["excerpt"]
        eid=ctx["hippocampus"].encode(excerpt,source="library:"+str(spec["item_id"]),
              kind="reading",epistemic_status="source_passage",
              generation=ctx["generation"],salience=.75,
              concepts=terms(str(spec.get("title",""))+" "+excerpt,16),
              payload={"item_id":spec["item_id"],"offset":record["offset"],
                       "next_offset":record["next_offset"]})["id"]
        # Semantic learning extracts attributable sentences and cross-book comparisons;
        # the raw excerpt remains an episode, not a confirmed scientific claim.
        insights=ctx["semantics"].observe({
            "id":eid,"text":excerpt,"epistemic_status":"source_passage",
            "source":"library:"+str(spec["item_id"]),
            "payload":{"title":spec.get("title","")}},
            generation=ctx["generation"])
        active=ctx["cortex"].activate(
           terms(str(spec.get("title",""))+" "+excerpt,13),strength=.76)
        ctx["cortex"].learn(active,reward=.13,source_event=eid)
        # Advancement and meaningful comparisons, not a fixed .55, determine
        # progress; an ordinary source read is still a real learning event.
        new_comparisons=sum(1 for insight in insights
                            if insight.get("kind")=="cross_source_comparison")
        progress=min(.9,.20+.16*min(3,new_comparisons)
                     +(.1 if record["next_offset"] is None else 0))
        ctx["curiosity"].notice(str(spec.get("title","")),novelty=.4,
               progress=progress,source=eid,generation=ctx["generation"])
        return Result("read",True,"Read "+str(len(excerpt))+" source characters",
               [eid],payload={"title":spec.get("title"),"next_offset":record["next_offset"],
                              "insights":len(insights)})

    def consolidate(spec,ctx):
        mode=str(spec.get("mode","nrem"))
        outcome=(ctx["consolidator"].rem(generation=ctx["generation"])
                 if mode=="rem" else ctx["consolidator"].nrem(generation=ctx["generation"]))
        return Result("consolidate",True,mode+" consolidation",payload=outcome)
    def connect(spec,ctx):
        cues=spec.get("cues") or []
        matches=ctx["cortex"].recall(cues)
        if len(matches)<2:return Result("connect",False,"Insufficient evidence for a new connection")
        left,right=matches[0],matches[1]
        ctx["cortex"].connect(left["id"],right["id"],relation="analogous_to",weight=.09)
        return Result("connect",True,"Formed tentative analogy",payload={"analogy":[left,right]})
    def reflect(spec,ctx):
        cues=spec.get("cues") or []
        associations=ctx["cortex"].recall(cues,limit=12)
        if not associations:return Result("explore",False,"No meaningful associations activated")
        active=ctx["cortex"].activate([a["label"] for a in associations[:7]],strength=.5)
        update=ctx["cortex"].learn(active,reward=0,rate=.01)
        return Result("explore",True,"Recombined existing memories",
                      payload={"associations":associations,"plasticity":update})
    def note(spec,ctx):
        text=str(spec.get("text","")).strip()
        if not text:return Result("note",False,"Empty note")
        ep=ctx["hippocampus"].encode(text,source="ubique:note",kind="thought",
                epistemic_status="model_interpretation",
                generation=ctx["generation"])
        return Result("note",True,"Note stored",[ep["id"]])
    def library(spec,ctx):
        action=str(spec.get("action","")).lower()
        if action not in LIBRARY_ACTIONS:return Result("library",False,"Invalid library command")
        if action=="read":return read({**spec},ctx)
        output=apply_library_action(spec,actor="ubique")
        return Result("library",True,action,payload={"result":str(output)[:600]})
    for name,func in {"read":read,"consolidate":consolidate,"connect":connect,
             "explore":reflect,"attend":reflect,"note":note,"library":library}.items():
        reg.register(name,func)
    return reg

def select(candidates:list[Candidate],curiosity:Curiosity,
           history:list[str])->Candidate:
    if not candidates:return Candidate("consolidate","No candidates",{"mode":"nrem"})
    last=history[-8:]
    for item in candidates:
        item.score-=.07*last.count(item.kind)
        if item.kind=="read":
            item.score+=.15*curiosity.score(str(item.payload.get("title","")))
        if item.kind=="rest":
            item.score-=.1
    return max(candidates,key=lambda x:x.score)
