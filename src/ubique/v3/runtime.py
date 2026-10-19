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
from .text_quality import content_terms, clean_source
from .workspace import Workspace
from .planner import Planner
from .self_model import SelfContinuity
from .library import KnowledgeLibrary
from .evolution import Evolution
from .semantics import SemanticLearner
from .inquiry import enquire
from .experimentation import run_local_experiment
from .evidence import assess, is_system_topic

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
        self.semantics=SemanticLearner(self.journal,self.hippocampus,
                                        self.world,self.cortex)
        self.consolidator=Consolidator(self.journal,self.hippocampus,
                                       self.cortex,self.world)
        self.registry=build_registry()
        self.workspace=Workspace(7)
        self.planner=Planner(self.journal)
        self.identity=SelfContinuity(self.journal)
        self.library=KnowledgeLibrary(self.curiosity)
        self.evolution=Evolution(self.journal,config)
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
            if is_system_topic(str(e.get("source",""))+" "+str(e.get("text",""))[:250]):
                continue
            for c in e.get("concepts",[])[:5]:
                if len(str(c))>=4 and not is_system_topic(c):
                    candidates.append(str(c))
        for item in self.curiosity.interests.values():
            if not is_system_topic(item["topic"]):
                candidates.append(item["topic"])
        # Bias scarce conscious attention towards actual recent reading,
        # without deleting unrelated or older memories from the cortex.
        recent_sources=[e for e in reversed(self.hippocampus.all_episodes())
                        if e.get("epistemic_status")=="source_passage"
                        and str(e.get("source","")).startswith("library:")][:4]
        source_terms=set()
        for episode in recent_sources:
            source_terms.update(content_terms(str(episode.get("text",""))[:3500],60))
            source_terms.update(content_terms(str(episode.get("payload",{}).get("title","")),12))
        # Contextual cortical reactivation: genuine current book passages
        # need to become active *before* competing with residual old issue nodes.
        # Limited to a few cues to preserve free association and memory variety.
        book_cues=[]
        for episode in recent_sources[:2]:
            book_cues+=content_terms(str(episode.get("payload",{}).get("title","")),8)
            book_cues+=content_terms(str(episode.get("text",""))[:1800],14)
        cues=[c for c in dict.fromkeys(book_cues) if not is_system_topic(c)][:18]
        if cues:
            self.cortex.activate(cues,strength=.46)
        workspace=self.workspace.compete(self.cortex,interest={
            key:float(data.get("interest",0)) for key,data in self.curiosity.interests.items()},
            grounding=source_terms)
        candidates.extend([item.label for item in workspace
                           if not is_system_topic(item.label)])
        books=[x.payload["title"] for x in reading_candidates(self.curiosity)]
        focus=self.curiosity.select(candidates,generation=generation)
        # A stray token extracted from boilerplate must not eclipse real books.
        if books and (not focus or
            (len(focus.split())==1 and self.curiosity.score(focus)<.5)):
            focus=self.curiosity.select(books,generation=generation)
        # Prefer genuine passages, not solely old issue/diagnostic chatter.
        passages=recent_sources[:3]
        recalled=list({e.get("id",str(i)):e for i,e in enumerate(passages+old)}.values())
        return focus,recalled[:10]

    def _provider_packet(self,generation:int,focus:str,recall:list[dict])->dict:
        if self.providers is None or not self.providers.available():
            return {"provider":None,"actions":[],"questions":[],"associations":[]}
        # LLMs are scarce cognitive collaborators, never the per-heartbeat clock.
        # A newly available provider gets an early opportunity.
        prev=read_json("v3.json",{})
        last=int(prev.get("last_provider_generation",-100) or -100)
        if generation-last<6 and generation%17!=0:
            return {"provider":None,"actions":[],"questions":[],"associations":[]}
        stances=self.identity.stances(self.world.stances)
        associations=self.cortex.recall([focus],limit=10)
        prompt=make_prompt(focus=focus,associations=associations,
            interests=list(self.curiosity.interests.values())[-12:],
            episodes=recall,known_stances=stances)
        self._packet_recall=recall[:6]
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
        # Only episodes actually provided as evidence to this contributor qualify.
        actual={str(ep.get("id")):ep for ep in getattr(self,"_packet_recall",[])
                if ep.get("id") and ep.get("epistemic_status") in
                {"source_passage","observed","observed_action_outcome"}}
        evidence=set(actual)
        for proposal in packet.get("world_model_updates",[])[:5]:
            if not isinstance(proposal,dict):continue
            statement=str(proposal.get("statement","")).strip()
            ids=list(dict.fromkeys(x for x in proposal.get("evidence_ids",[])
                                    if x in evidence))
            assessment=assess(statement,[actual[x] for x in ids])
            self.journal.append("evidence_review",{
                "statement":statement[:1200],"evidence_ids":ids,
                "assessment":assessment.as_dict(),"provider":packet.get("provider")},
                generation=generation)
            if statement:
                accepted=assessment.verdict in {"plausible","directly_supported"}
                self.world.claim(statement,
                    status="INFERRED" if accepted else "HYPOTHESIS",
                    source_refs=ids,confidence=min(.38,float(proposal.get("confidence",.4)))
                    if accepted else .12,generation=generation)
        for stance in packet.get("stance_updates",[])[:3]:
            ids=list(dict.fromkeys(e for e in stance.get("evidence_ids",[])
                                    if e in evidence))
            if not ids:
                continue
            position=str(stance.get("position",""))
            assessment=assess(position,[actual[e] for e in ids])
            self.journal.append("evidence_review",{
                "statement":position[:1200],"evidence_ids":ids,
                "assessment":assessment.as_dict(),"stance_topic":stance.get("topic"),
                "provider":packet.get("provider")},generation=generation)
            if assessment.verdict in {"plausible","directly_supported"}:
                self.world.stance(str(stance["topic"]),position,
                    evidence_ids=ids,relation=str(stance.get("relation","new")),
                    uncertainty=max(.6,1-float(stance.get("confidence",.5))),
                    generation=generation)
            else:
                # An imaginative interpretation is welcome; an unsupported
                # interpretation must not be silently installed as identity.
                self.world.claim("Proposed view: "+position,status="HYPOTHESIS",
                                 source_refs=ids,confidence=.12,generation=generation)
        return candidates

    def _candidates(self,generation:int,focus:str,recall:list[dict],
                    model:list[Candidate])->list[Candidate]:
        candidates=reading_candidates(self.curiosity)
        # Sustain one line of reading long enough to connect consecutive
        # chapters, then give the other books room again.
        read_history=[e.get("data",{}) for e in self.journal.events()
                      if e["kind"]=="action_outcome"
                      and e.get("data",{}).get("kind")=="read"
                      and e.get("data",{}).get("success")][-8:]
        if read_history:
            last_title=str(read_history[-1].get("payload",{}).get("title",""))
            last_streak=0
            for item in reversed(read_history):
                if item.get("payload",{}).get("title")==last_title:
                    last_streak+=1
                else:
                    break
            for candidate in candidates:
                if candidate.payload.get("title")==last_title:
                    candidate.score+=(.21 if last_streak<4 else -.22)
        candidates+=model
        # Autonomous experiments and questions compete naturally with reading.
        # They work offline and do not need a predetermined philosophical answer.
        if generation % 11 == 0 or "experiment" not in self.history[-40:]:
            candidates.append(Candidate("experiment",
                "Investigate an uncertain association using local memory",
                {"experiment_type":"association_probe","hypothesis":focus},
                .73, source="curiosity"))
        if generation % 13 == 0 or "question" not in self.history[-40:]:
            candidates.append(Candidate("question",
                "Compare distinct books and leave an open question",
                {"topic":focus}, .69, source="curiosity"))
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
        from .research import discover
        if not topic:return Result("search",False,"No topic available")
        previous=read_json("v3.json",{})
        last=int(previous.get("last_search_generation",-100) or -100)
        if generation-last<20:
            return Result("search",True,"Waiting before next external source request")
        try:
            results=discover(topic)
        except Exception as exc:
            return Result("search",False,type(exc).__name__+": "+str(exc)[:200])
        seen=[]
        for item in results:
            ep=self.hippocampus.encode(item["text"],source=item["url"],
                    kind="reading",epistemic_status="source_passage",
                    generation=generation,salience=.65,
                    concepts=terms(item["title"]+" "+item["text"],18),
                    payload={"title":item["title"],"license":item["attribution"]})
            self.semantics.observe(ep,generation=generation)
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
             "world":self.world,"consolidator":self.consolidator,
             "semantics":self.semantics}
        if chosen.kind=="rest":
            self.cortex.step(decay=.6)
            return Result("rest",True,"Local activation decayed")
        if chosen.kind=="search":
            return self._research(focus,generation)
        if chosen.kind in {"evolve","edit_code"}:
            if not self.providers or not self.providers.available():
                return Result(chosen.kind,False,"Remote code contributor deferred")
            outcome=self.evolution.attempt(chosen.reason,generation=generation,
                                           target=str(chosen.payload.get("target","")))
            return Result(chosen.kind,outcome["success"],outcome["result"],
                          provider=outcome["provider"])
        if chosen.kind=="question":
            inquiry=enquire(self.hippocampus,self.journal,self.world,
                            generation=generation)
            if inquiry:
                return Result("question",True,inquiry["question"],
                              inquiry["source_ids"],payload=inquiry)
            return Result("question",False,"No new source pair to compare")
        if chosen.kind=="experiment":
            kind=str(chosen.payload.get("experiment_type",""))
            from ..experiments import SAFE_EXPERIMENTS,run_experiment
            # External providers are optional; local experiments are first-class.
            if (chosen.payload.get("remote") and kind in SAFE_EXPERIMENTS
                    and self.providers and self.providers.available()):
                outcome=run_experiment(kind,str(chosen.payload.get("target","")),
                     self.providers.router,str(chosen.payload.get("hypothesis","")))
                return Result("experiment",bool(outcome.get("passed")),str(outcome)[:800],
                              provider=str(outcome.get("provider","remote")),
                              payload=outcome)
            outcome=run_local_experiment(chosen.payload,ctx)
            return Result("experiment",bool(outcome["performed"]),
                          str(outcome.get("status","inconclusive")),
                          outcome.get("evidence_ids",[]),provider="local",
                          payload=outcome)
        if chosen.kind=="library" and chosen.payload.get("action")=="request":
            if not self.library.can_request(str(chosen.payload.get("title",""))):
                return Result("library",True,"Equivalent reading wish already exists")
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
                          "interests":len(self.curiosity.interests),
                          "associative_comparisons":len(self.semantics.comparisons),
                          "screened_interpretations":sum(1 for e in self.journal.events()
                              if e["kind"]=="evidence_review" and
                              e["data"].get("assessment",{}).get("verdict")
                              in {"plausible","directly_supported"}),
                          "historical_insight_labels":sum(1 for e in self.journal.events()
                              if e["kind"]=="insight")},
                "journal":{"events":self.journal.manifest["events"],
                           "segments":len(self.journal._files()),
                           "compressed_snapshot":self.journal.snapshot_path.is_file(),
                           "snapshot_events":self.journal.manifest["snapshot_events"]},
                "migration":self.migration.get("loaded_from"),
                "evidence_reviews":sum(1 for e in self.journal.events()
                                      if e["kind"]=="evidence_review"),
                "workspace":self.workspace.snapshot(),
                "skills":self.identity.skills(),
                "active_stances":len(self.identity.stances(self.world.stances))}
        write_json("v3.json",status)
        active=result.success and chosen.kind not in {"rest","consolidate","question"}
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
            # Code changes are a legitimate action when observed failures recur.
            # They compete with reading and thought rather than being mandatory.
            failures=sum(1 for kind in ("search","connect","experiment","read")
                         if self.planner.quality[kind][1]>=3
                         and self.planner.quality[kind][0]/self.planner.quality[kind][1]<.85)
            if self.providers and self.providers.available() and (failures or
                  (len(self.semantics.comparisons)>=12 and generation%79==0)):
                recent_changes=[e for e in self.journal.events()
                                if e["kind"]=="change_intent"][-1:]
                last_change=int(recent_changes[-1]["generation"] or 0) if recent_changes else -1000
                if generation-last_change>=72:
                    reason=("Investigate recurring failures in the v3 learning and action cycle."
                            if failures else "Examine what the recent cross-source learning has revealed.")
                    candidates.append(Candidate("evolve",
                        reason+" Propose one small reversible change to improve the curiosity, "
                        "reasoning or experiential learning process.",
                        {"cause":"observed_learning"},.84,
                        source="autonomous_observation"))
            candidates=self.planner.rank(candidates,last=self.history)
            chosen=select(candidates,self.curiosity,self.history)
            result=self._act(chosen,generation=generation,focus=focus)
            if not result.success and chosen.kind in {"search","read"}:
                # Preserve the failed attempt as experience, but never waste
                # the entire learning cycle on a blocked external source.
                self.journal.append("action_outcome",{
                    "kind":chosen.kind,"success":False,"reason":chosen.reason,
                    "effect":result.effect,"provider":result.provider},
                    generation=generation)
                others=[x for x in candidates if x is not chosen and
                        (x.kind!="read" or x.payload.get("item_id")!=chosen.payload.get("item_id"))]
                for alternative in others[:5]:
                    attempt=self._act(alternative,generation=generation,focus=focus)
                    if attempt.success:
                        chosen,result=alternative,attempt
                        break
            self.history.append(chosen.kind)
            self.planner.record(chosen.kind,result.success)
            self.identity.experience(chosen.kind,outcome=result.success,
                                      generation=generation,reason=chosen.reason)
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
