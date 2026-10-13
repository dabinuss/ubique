"""Fast episodic encoding, temporal retrieval and balanced replay."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Iterable
from ..state import ROOT
from .storage import Journal

def terms(text: str, limit=16) -> list[str]:
    stop={"that","this","with","from","about","would","could","when","then","which",
          "there","where","their","these","haben","wurde","einer","einen","einem",
          "nicht","auch","diese","dass","sowie","between","human","study"}
    words=re.findall(r"[^\W\d_]{4,}",text.casefold(),flags=re.UNICODE)
    return list(dict.fromkeys(w for w in words if w not in stop))[:limit]

class Hippocampus:
    def __init__(self, journal: Journal, *, legacy_path: Path | None = None):
        self.journal=journal
        self.legacy_path=legacy_path or ROOT/"memory"/"brain_episodes.jsonl"
        self._legacy: list[dict] | None=None

    def legacy(self) -> list[dict]:
        if self._legacy is None:
            if not self.legacy_path.exists():
                self._legacy=[]
            else:
                records=[]
                for line in self.legacy_path.read_text(encoding="utf-8").splitlines():
                    try: v=json.loads(line)
                    except json.JSONDecodeError:continue
                    if isinstance(v,dict):records.append(v)
                self._legacy=records
        return self._legacy

    def all_episodes(self) -> list[dict]:
        old=list(self.legacy())
        new=[{**event["data"], "id": event["id"], "timestamp": event["timestamp"]}
             for event in self.journal.events() if event.get("kind")=="episode"]
        return old+new

    def encode(self, text: str, *, source: str, kind="experience",
               epistemic_status="observed_action_outcome", generation=None,
               salience=0.5, concepts: Iterable[str] | None=None, payload=None) -> dict:
        if not text.strip():
            raise ValueError("Cannot encode an empty experience")
        episode={"kind":kind,"text":text[:12000],"source":source[:350],
                 "concepts":list(dict.fromkeys(concepts or terms(text)))[:24],
                 "epistemic_status":epistemic_status,"salience":float(salience),
                 "generation":generation,"payload":payload or {}}
        ev=self.journal.append("episode",episode,generation=generation)
        episode.update({"id":ev["id"],"timestamp":ev["timestamp"]})
        return episode

    def recall(self, query: str | Iterable[str], limit=10) -> list[dict]:
        keys=set(terms(query if isinstance(query,str) else " ".join(query),24))
        if not keys:
            return self.all_episodes()[-limit:]
        records=self.all_episodes()
        scored=[]
        for age,ep in enumerate(reversed(records)):
            concepts=set(str(x).casefold() for x in ep.get("concepts",[]))
            words=set(terms(str(ep.get("text",""))[:1800],40))
            overlap=2.0*len(concepts&keys)+len(words&keys)
            if overlap:
                scored.append((overlap+float(ep.get("salience",0.3))*0.1
                               +0.08/(1+age/30),ep))
        return [ep for _,ep in sorted(scored,key=lambda s:s[0],reverse=True)[:limit]]

    def replay(self, limit=10, seed=0) -> list[dict]:
        """Interleave old/new and different thematic roots, not just latest items."""
        records=self.all_episodes()
        if not records:return []
        stride=max(1,len(records)//max(4,limit*2))
        pool=records[seed%stride::stride]+records[-limit*2:]
        by_topic:dict[str,list[dict]]={}
        for ep in pool:
            label=next(iter(ep.get("concepts",[]) or ["other"]),"other")
            by_topic.setdefault(str(label),[]).append(ep)
        chosen=[];grouped=sorted(by_topic.values(),key=lambda v:len(v))
        for entries in grouped:
            chosen.append(entries[(seed+len(chosen))%len(entries)])
            if len(chosen)>=limit:break
        for ep in reversed(records):
            if len(chosen)>=limit:break
            if ep not in chosen:chosen.append(ep)
        return chosen

    def count(self) -> int:
        return len(self.all_episodes())
