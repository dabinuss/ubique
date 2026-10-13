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
        if response.status_code in {403,404,429}:
            return []  # Source denied; local reading continues instead.
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


def openalex(topic: str) -> list[dict]:
    """Retrieve a small number of *actual abstracts* with explicit work URLs.

    Do not turn bare titles or search-result descriptions into read passages.
    """
    query = topic.strip()[:120]
    if not query:
        return []
    with httpx.Client(timeout=12, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get("https://api.openalex.org/works", params={
            "search": query, "filter": "has_abstract:true",
            "per-page": 3, "select": "id,display_name,abstract_inverted_index,doi"})
        if response.status_code in {403, 404, 429}:
            return []
        response.raise_for_status()
        items = response.json().get("results", [])
    out = []
    for item in items:
        inverted = item.get("abstract_inverted_index") or {}
        if not isinstance(inverted, dict) or not inverted:
            continue
        indexed = [(int(position), word) for word, positions in inverted.items()
                   for position in positions if isinstance(position, int)]
        indexed.sort(key=lambda value: value[0])
        content = " ".join(word for _, word in indexed)[:3500]
        url = item.get("doi") or item.get("id")
        if not content or not url:
            continue
        out.append({"title": str(item.get("display_name", ""))[:250],
                    "text": content, "url": str(url),
                    "source": "OpenAlex scholarly work abstract",
                    "attribution": "Abstract metadata indexed by OpenAlex",
                    "retrieval_kind": "source_passage"})
    return out


def discover(topic: str) -> list[dict]:
    """Try different verifiable sources, with no invented replacement passages."""
    try:
        pages = wikipedia(topic)
        if pages:
            return pages
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    try:
        return openalex(topic)
    except (httpx.HTTPError, ValueError, KeyError):
        return []
