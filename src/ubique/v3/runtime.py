"""Ubique 3: persistent autonomous local cognition with optional language support."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from ..config import Config
from ..github import GitHubClient
from ..state import ROOT, start_cycle, finish_cycle, write_json, utc_now, read_json
from ..brain.runtime import NeurocognitiveRuntime
from ..brain.episodic import extract_concepts
from .storage import Journal
from .migration import bootstrap, continuity
from .hippocampus import Hippocampus, terms
from .curiosity import Curiosity
from .representation import Cortex
from .world_model import WorldModel
from .consolidation import Consolidator
from .actions import Candidate, Result, build_registry, reading_candidates, select
from .providers import ThinkingProviders, make_prompt

log=logging.getLogger("ubique.v3")
MAX_CANDIDATES=20

class UbiqueV3:
    def __init__(self, config:Config, *, storage_root:Path|None=None,
                 enable_remote:bool=True):
        self.config=config
        self.journal=Journal(storage_root)
        self.cortex,self.migration=bootstrap(self.journal)
        self.hippocampus=Hippocampus(self.journal)
        self.world=WorldModel(self.journal)
        self.curiosity=Curiosity(self.journal)
        self.consolidator=Consolidator(self.journal,self.hippocampus,
                                       self.cortex,self.world)
        self.registry=build_registry()
        self.providers=ThinkingProviders(config) if enable_remote else None
        self.github=GitHubClient(config.github_token,config.github_repository)
        self.last_packet=None
        self.since_snapshot=int(self.journal.manifest.get("events",0)
                                -self.journal.manifest.get("snapshot_events",0))
        self.history=[e.get("data",{}).get("kind","") for e in
                      self.journal.events() if e.get("kind")=="action_outcome"][-30:]

    def _recent_focus(self,generation:int)->tuple[str,list[dict]]:
        old=[e for e in self.hippocampus.replay(10,generation)
             if e.get("concepts") or e.get("text")]
        candidates=[]
        for e in old:
            for c in e.get("concepts",[])[:5]:
                if len(str(c))>=4:candidates.append(str(c))
        for item in self.curiosity.interests.values():
            candidates.append(item["topic"])
        # Adopt topics learned from actual sources, avoiding fixed philosophy goals.
        focus=self.curiosity.select(candidates,generation=generation)
        return focus,old

    def _provider_packet(self,generation:int,focus:str,recall:list[dict])->dict:
        if self.providers is None or not self.providers.available():
            return {"provider":None,"actions":[],"questions":[],"associations":[]}
        # LLMs are scarce cognitive collaborators, never the per-heartbeat clock.
        # A newly available provider gets an early opportunity.
        prev=read_json("v3.json",{})
        last=int(prev.get("last_provider_generation",-100) or -100)
        if generation-last<6 and generation%17!=0:
            return {"provider":None,"actions":[],"questions":[],"associations":[]}
        stances=read_json("brain.json",{}).get("current_stances",[])
        associations=self.cortex.recall([focus],limit=10)
        prompt=make_prompt(focus=focus,associations=associations,
            interests=list(self.curiosity.interests.values())[-12:],
            episodes=recall,known_stances=stances)
        return self.providers.think(prompt)

    def _handle_model(self,packet:dict,generation:int)->list[Candidate]:
        if not packet.get("provider"):return []
        candidates=[]
        all_concepts=[]
        for obj in packet.get("associations",[])[:10]:
            label=str(obj.get("label","")).strip()
            if label:all_concepts.append(label)
        active=self.cortex.activate(all_concepts,strength=.45)
        if active:self.cortex.learn(active,rate=.012,source_event="model:"+str(generation))
        for question in packet.get("questions",[])[:4]:
            self.hippocampus.encode(str(question),source="provider:"+str(packet["provider"]),
                kind="question",epistemic_status="model_question",generation=generation,
                salience=.45)
        for hypothesis in packet.get("hypotheses",[])[:3]:
            self.world.claim(str(hypothesis),status="HYPOTHESIS",
                source_refs=[],confidence=.18,generation=generation)
        for spec in packet.get("actions",[])[:5]:
            kind=str(spec.get("kind","")).strip().lower()
            if kind=="library" and (spec.get("payload") or {}).get("action")=="read":
                kind="read"
            if kind not in {"read","library","consolidate","explore","connect",
                            "experiment","evolve","edit_code","note"}:continue
            payload=spec.get("payload") if isinstance(spec.get("payload"),dict) else {}
            if kind=="read" and not payload.get("item_id"):continue
            score=(.35*float(spec.get("support",.5))+
                   .35*float(spec.get("information_gain",.4))+
                   .3*float(spec.get("utility",.5)))
            candidates.append(Candidate(kind,str(spec.get("description",""))[:600],
                                        payload,score,source="model"))
        # Do not confuse model-proposed stances with observed evidence.
        evidence=set(ep.get("id") for ep in self.hippocampus.recall(
            " ".join(all_concepts),limit=12))
        for stance in packet.get("stance_updates",[])[:3]:
            ids=[e for e in stance.get("evidence_ids",[]) if e in evidence]
            if ids:
                self.world.stance(str(stance["topic"]),str(stance["position"]),
                    evidence_ids=ids,relation=str(stance.get("relation","new")),
                    uncertainty=1-float(stance.get("confidence",.5)),
                    generation=generation)
        return candidates

    def _candidates(self,generation:int,focus:str,recall:list[dict],
                    model:list[Candidate])->list[Candidate]:
        candidates=reading_candidates(self.curiosity)
        candidates+=model
        active=self.cortex.recall([focus])
        if active:
            candidates.append(Candidate("explore","Follow associative paths from "+focus,
                 {"cues":[focus]},.53+self.curiosity.score(focus)*.20))
            if len(active)>=2:
                candidates.append(Candidate("connect","Compare related memories",
                    {"cues":[focus]},.49))
        candidates.append(Candidate("consolidate","Revisit memories from different periods",
                      {"mode":"nrem"},.53 if self.history[-2:]!=["consolidate"]*2 else .2))
        candidates.append(Candidate("consolidate","Explore novel analogies offline",
                      {"mode":"rem"},.34))
        # Rest is allowed but never the sole response to missing API keys.
        candidates.append(Candidate("rest","Reduce activation",{},.13))
        if not candidates or not focus:
            candidates.append(Candidate("explore","Explore past experiences",
                                {"cues":terms(str(recall[0].get("text",""))) if recall else []},.45))
        return candidates[:MAX_CANDIDATES]

    def _research(self,topic:str,generation:int)->Result:
        from .research import wikipedia
        if not topic:return Result("search",False,"No topic available")
        previous=read_json("v3.json",{})
        last=int(previous.get("last_search_generation",-100) or -100)
        if generation-last<20:
            return Result("search",True,"Waiting before next external source request")
        try:
            results=wikipedia(topic)
        except Exception as exc:
            return Result("search",False,type(exc).__name__+": "+str(exc)[:200])
        seen=[]
        for item in results:
            ep=self.hippocampus.encode(item["text"],source=item["url"],
                    kind="reading",epistemic_status="source_passage",
                    generation=generation,salience=.65,
                    concepts=terms(item["title"]+" "+item["text"],18),
                    payload={"title":item["title"],"license":item["attribution"]})
            self.world.claim(item["text"][:750],status="SOURCE_CLAIM",
                             source_refs=[ep["id"]],confidence=.3,generation=generation)
            active=self.cortex.activate(terms(item["title"]+" "+item["text"],12),
                                        strength=.65)
            self.cortex.learn(active,reward=.1,source_event=ep["id"])
            self.curiosity.notice(item["title"],novelty=.7,progress=.5,
                                  source=ep["id"],generation=generation)
            seen.append(ep["id"])
        return Result("search",bool(seen),"Acquired "+str(len(seen))+
                      " attributable external articles",seen)

    def _act(self,chosen:Candidate,*,generation:int,focus:str)->Result:
        ctx={"generation":generation,"hippocampus":self.hippocampus,
             "cortex":self.cortex,"curiosity":self.curiosity,
             "world":self.world,"consolidator":self.consolidator}
        if chosen.kind=="rest":
            self.cortex.step(decay=.6)
            return Result("rest",True,"Local activation decayed")
        if chosen.kind=="search":
            return self._research(focus,generation)
        if chosen.kind in {"evolve","edit_code"}:
            if not self.providers or not self.providers.available():
                return Result(chosen.kind,False,"Remote code contributor deferred")
            from ..brain.runtime import NeurocognitiveRuntime
            ok,text,provider=NeurocognitiveRuntime(self.config)._evolve(
                chosen.reason,generation)
            return Result(chosen.kind,ok,text,provider=provider)
        if chosen.kind=="experiment":
            kind=str(chosen.payload.get("experiment_type",""))
            from ..experiments import SAFE_EXPERIMENTS,run_experiment
            if kind not in SAFE_EXPERIMENTS or self.providers is None:
                return Result("experiment",False,"Unsupported or offline experiment")
            outcome=run_experiment(kind,str(chosen.payload.get("target","")),
                 self.providers.router,str(chosen.payload.get("hypothesis","")))
            return Result("experiment",bool(outcome.get("passed")),str(outcome)[:800],
                          provider=str(outcome.get("provider","deterministic")))
        return self.registry.execute(chosen,ctx)

    def _persist(self,generation:int,result:Result,chosen:Candidate,
                 packet:dict,focus:str)->None:
        # Compress repeated upserts within the same pulse into the last state.
        delta={}
        for change in self.cortex.changes:
            d=change["data"]
            key=change["kind"]+":"+(d.get("id") or
                 (d.get("source","")+"|"+d.get("target","")+"|"+d.get("relation","")))
            delta[key]=change
        delta["tick"]={"kind":"cortex_tick","data":{"tick":self.cortex.tick}}
        self.journal.append("graph_delta",{"changes":list(delta.values())},generation=generation)
        self.cortex.changes.clear()
        if self.journal.manifest["events"]-self.journal.manifest["snapshot_events"]>=80:
            self.journal.snapshot(self.cortex.as_dict())
        status={"version":3,"architecture":"neurocognitive-v3",
                "timestamp":utc_now(),"generation":generation,"focus":focus,
                "last_action":chosen.kind,"last_action_success":result.success,
                "last_action_detail":result.effect[:400],
                "last_provider":packet.get("provider"),
                "last_provider_generation":generation if packet.get("provider")
                     else read_json("v3.json",{}).get("last_provider_generation",-100),
                "last_search_generation":generation if chosen.kind=="search"
                     else read_json("v3.json",{}).get("last_search_generation",-100),
                "cortex":{"nodes":len(self.cortex.nodes),
                          "synapses":len(self.cortex.edges),
                          "active":sum(n.activation>.05 for n in self.cortex.nodes.values()),
                          "archived_synapses":sum(e.archived for e in self.cortex.edges.values())},
                "memory":{"episodes":self.hippocampus.count(),
                          "claims":len(self.world.claims),
                          "stances":len(self.world.stances),
                          "interests":len(self.curiosity.interests)},
                "journal":{"events":self.journal.manifest["events"],
                           "snapshot_events":self.journal.manifest["snapshot_events"]},
                "migration":self.migration.get("loaded_from")}
        write_json("v3.json",status)
        active=result.success and chosen.kind not in {"rest","consolidate"}
        write_json("pulse.json",{"timestamp":utc_now(),"generation":generation,
              "mode":"active" if active else "quiet","should_continue":active,
              "minimum_delay_seconds":60,
              "reason":"v3:"+chosen.kind,"selected_action":chosen.kind,
              "workspace_activation":status["cortex"]["active"]/120,
              "watchdog_schedule":"3,18,33,48 * * * *"})

    def run(self)->int:
        # Preserve mature external-issue handling from v2 while migrating
        # its behavior in the rest of the autonomous cycles.
        if self.github.enabled:
            try:
                if self.github.list_tasks(self.config.max_tasks):
                    log.info("Delegating existing GitHub issue task to proven v2 handler")
                    return NeurocognitiveRuntime(self.config).run()
            except Exception:
                log.exception("External issue polling failed; local brain continues")
        state=start_cycle()
        generation=int(state["generation"])
        try:
            focus,recall=self._recent_focus(generation)
            if focus:
                self.curiosity.notice(focus,novelty=.22,uncertainty=.45,
                     source="associative_recall",generation=generation)
                active=self.cortex.activate([focus],strength=.7)
                if active:self.cortex.learn(active,reward=0.0,rate=.006)
            packet=self._provider_packet(generation,focus,recall)
            self.last_packet=packet
            model=self._handle_model(packet,generation)
            self.cortex.step()
            candidates=self._candidates(generation,focus,recall,model)
            # Exploration can broaden knowledge, not repeat identical wishes.
            if focus and (generation%23==0 or not reading_candidates(self.curiosity)):
                # Periodically seek knowledge beyond the initial book collection.
                last_search=int(read_json("v3.json",{}).get("last_search_generation",-100) or -100)
                if generation-last_search>=20:
                    candidates.append(Candidate("search","Acquire new general knowledge about "+focus,
                              {"topic":focus},.76))
            chosen=select(candidates,self.curiosity,self.history)
            result=self._act(chosen,generation=generation,focus=focus)
            self.history.append(chosen.kind)
            event=self.journal.append("action_outcome",{
                "kind":chosen.kind,"success":result.success,
                "provider":result.provider,"reason":chosen.reason,
                "effect":result.effect,"evidence_ids":result.evidence_ids,
                "payload":result.payload},generation=generation)
            if chosen.kind not in {"rest","consolidate"}:
                self.cortex.feedback([n.id for n in
                    sorted(self.cortex.nodes.values(),
                           key=lambda n:n.activation,reverse=True)[:6]],
                    1 if result.success else -0.5)
            self.cortex.homeostasis()
            self._persist(generation,result,chosen,packet,focus)
            finish_cycle(state,"v3:action="+chosen.kind+
                ":success="+str(result.success)+
                ":provider="+str(packet.get("provider"))+
                ":synapses="+str(len(self.cortex.edges)))
            log.info("Ubique v3 generation %s: %s %s; %s synapses",
                     generation,chosen.kind,result.success,len(self.cortex.edges))
            return 0
        except Exception as exc:
            finish_cycle(state,"v3_failed:"+type(exc).__name__)
            write_json("pulse.json",{"timestamp":utc_now(),
                "generation":generation,"should_continue":False,
                "mode":"quiet","reason":"v3_failed:"+type(exc).__name__})
            log.exception("V3 cycle failed")
            return 2
