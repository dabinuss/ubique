"""Autonomous interest formation with saturation and cross-domain exploration."""
from __future__ import annotations
import math
import re
from typing import Iterable
from .storage import Journal

def normalized(text:str) -> set[str]:
    stop={"of","the","and","between","human","during","direct","measurements","quantitative",
          "empirical","an","in","with","on","a","to","for","study","review","of","der",
          "die","das","und","von","eine","einer","zum"}
    return {w for w in re.findall(r"[a-zäöüß]{3,}",text.casefold()) if w not in stop}

def related(a:str,b:str) -> float:
    wa,wb=normalized(a),normalized(b)
    return len(wa&wb)/max(1,len(wa|wb))

class Curiosity:
    def __init__(self,journal:Journal):
        self.journal=journal
        self.interests={}
        self.wishes=[]
        for ev in journal.events():
            if ev["kind"]=="interest":
                d=ev["data"];self.interests[d["topic"]]=d
            if ev["kind"]=="wish":
                self.wishes.append(ev["data"])

    def notice(self,topic:str,*,novelty=.4,progress=.0,uncertainty=.5,
               source="",generation=None):
        topic=" ".join(topic.split())[:160]
        if not topic:return {}
        previous=self.interests.get(topic,{})
        visits=int(previous.get("visits",0))+1
        saturation=max(0.0,0.09*(visits-2))
        record={"topic":topic,"visits":visits,"novelty":float(novelty),
                "progress":float(progress),"uncertainty":float(uncertainty),
                "saturation":min(1,saturation),
                "interest":max(0.0,min(1.0,
                    0.30*novelty+0.35*max(0,progress)+0.35*uncertainty
                    -saturation+float(previous.get("interest",0))*0.25)),
                "source":source}
        self.journal.append("interest",record,generation=generation)
        self.interests[topic]=record
        return record

    def score(self,topic:str,*,bridge=.0) -> float:
        known=self.interests.get(topic,{})
        return (known.get("interest",0.15)
                +0.25*bridge
                -0.35*known.get("saturation",0))

    def select(self,candidates:Iterable[str],*,generation=0) -> str:
        possibilities=list(dict.fromkeys(x for x in candidates if x))
        if not possibilities:return ""
        # Deterministic shift within similar scores, ensuring rare topics get airtime.
        return max(possibilities,key=lambda x:
                   self.score(x)+0.04*((hashlib_score(x,generation)%13)/13))

    def similar_wish(self,title:str,threshold=0.50) -> dict|None:
        for wish in self.wishes:
            if related(title,str(wish.get("title","")))>=threshold:
                return wish
        return None

    def wish(self,title:str,*,reason="",generation=None) -> dict:
        match=self.similar_wish(title)
        if match:return {**match,"already_requested":True}
        item={"title":title[:250],"reason":reason[:500],"already_requested":False}
        self.journal.append("wish",item,generation=generation)
        self.wishes.append(item)
        return item

def hashlib_score(text:str,generation:int)->int:
    import hashlib
    return int(hashlib.sha256((str(generation//6)+text).encode()).hexdigest()[:8],16)
