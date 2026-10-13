"""Bounded, attributed public-source discovery for self-directed general learning."""
from __future__ import annotations
from urllib.parse import urlencode
import httpx

USER_AGENT="UbiqueResearch/3.0 (educational autonomous research project)"
MAX_RESULTS=3

def wikipedia(topic:str,*,language="en") -> list[dict]:
    """Query encyclopedic public knowledge; passages remain source claims."""
    topic=topic.strip()[:120]
    if not topic:return []
    # One modest request, response bounded and no automatic source claims.
    url="https://"+language+".wikipedia.org/w/api.php"
    params={"action":"query","generator":"search","gsrsearch":topic,
            "gsrlimit":MAX_RESULTS,"prop":"extracts|info",
            "exintro":1,"explaintext":1,"inprop":"url","format":"json"}
    with httpx.Client(timeout=12,headers={"User-Agent":USER_AGENT}) as client:
        response=client.get(url,params=params)
        response.raise_for_status()
        payload=response.json()
    pages=(payload.get("query") or {}).get("pages",{})
    out=[]
    for item in list(pages.values())[:MAX_RESULTS]:
        title=str(item.get("title",""))
        text=str(item.get("extract","")).strip()
        if not title or not text:continue
        out.append({"title":title,"text":text[:5000],
                    "url":item.get("fullurl",""),
                    "source":"Wikipedia","attribution":"CC BY-SA / Wikipedia",
                    "retrieval_kind":"source_passage"})
    return out
