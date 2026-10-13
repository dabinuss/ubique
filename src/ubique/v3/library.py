"""Semantic wishes and source availability are separate from actual reading."""
from __future__ import annotations
from ..library import library_catalog
from .curiosity import Curiosity,related

class KnowledgeLibrary:
    def __init__(self,curiosity:Curiosity):
        self.curiosity=curiosity

    def available(self):
        return [item for item in library_catalog(250)
                if item.get("content_available") and not item.get("fully_read")]

    def find_request_family(self,title:str,threshold=.5):
        matches=[]
        for item in library_catalog(250):
            if item.get("kind")!="reading_request":continue
            score=related(str(item.get("title","")),title)
            if score>=threshold:matches.append((score,item))
        return sorted(matches,key=lambda p:p[0],reverse=True)

    def can_request(self,title:str)->bool:
        return not self.find_request_family(title)

    def explain_availability(self,item:dict)->str:
        if item.get("fully_read"):return "fully_read"
        if not item.get("content_available"):return "requested_not_acquired"
        return "readable"
