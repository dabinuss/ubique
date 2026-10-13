"""Budget-sensitive external thought contributors, optional to local cognition."""
from __future__ import annotations
from ..providers.gemini import GeminiProvider
from ..providers.groq import GroqProvider
from ..providers.huggingface import HuggingFaceProvider
from ..providers.router import ProviderRouter
from ..brain.substrate import CognitiveSubstrateManager
from ..config import Config

class ThinkingProviders:
    def __init__(self,config:Config):
        self.router=ProviderRouter([
            GeminiProvider(config.gemini_api_key,config.gemini_model),
            GroqProvider(config.groq_api_key,config.groq_model),
            HuggingFaceProvider(config.hf_token,config.hf_model,config.hf_endpoint),
        ],daily_limits={"gemini":config.gemini_daily_limit,
                        "groq":config.groq_daily_limit})
        self.parse=CognitiveSubstrateManager.parse_packet

    def available(self)->list[str]:
        return [n for n,d in self.router.remote_eligibility().items()
                if d.get("eligible")]

    def think(self,prompt:str,*,preferred:str|None=None)->dict:
        eligible=self.available()
        if not eligible:return {"provider":None,"actions":[],"questions":[],
                                 "associations":[],"reason":"no_eligible_provider"}
        name=preferred if preferred in eligible else eligible[0]
        try:
            result=self.router.generate_with(name,prompt[:22000])
            return self.parse(result.text,result.provider,result.model)
        except Exception as exc:
            return {"provider":None,"actions":[],"questions":[],"associations":[],
                    "reason":type(exc).__name__+":"+str(exc)[:300]}

def make_prompt(*,focus:str,associations:list[dict],interests:list[dict],
                episodes:list[dict],known_stances:list[dict])->str:
    import json
    context={"focus":focus,"associations":associations[:10],
             "interests":interests[:12],
             "episodes":[{"id":e.get("id"),"source":e.get("source"),
                          "text":str(e.get("text",""))[:800]} for e in episodes[:6]],
             "stances":known_stances[:8]}
    return ("You are a temporary language and reasoning contributor to Ubique 3. "
            "Ubique's persistent self and learning are in its own neural and semantic memory. "
            "Follow the currently interesting subject, including nonphilosophical subjects. "
            "Suggest connections, questions and genuine learnable actions, without inventing observations. "
            "For book reading use existing item ids supplied by Ubique. "
            "Return JSON containing associations, questions, hypotheses, stance_updates "
            "and actions. Actions should be read/library, experiment, evolve, connect, explore "
            "or consolidate; do not require self-optimization. Label hypotheses as uncertain. "
            "Context: "+json.dumps(context,ensure_ascii=False)[:18000])
