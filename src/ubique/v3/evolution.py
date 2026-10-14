"""Autonomous, auditable and reversible v3 self-evolution.

Change ideas come from lived learning / failed actions, never mandatory benchmarks.
Proposals use a small genuine source file as grounding; existing isolated test
and PR mechanisms remain the final execution/recovery path.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from ..brain.runtime import NeurocognitiveRuntime
from ..config import Config
from ..evolution import run_evolution
from ..state import ROOT
from .storage import Journal

EVOLVABLE = (
    "src/ubique/v3/curiosity.py",
    "src/ubique/v3/planner.py",
    "src/ubique/v3/workspace.py",
    "src/ubique/v3/semantics.py",
)

class Evolution:
    def __init__(self, journal: Journal, config: Config):
        self.journal=journal
        self.config=config

    def attempt(self, reason: str, *, generation: int):
        target=EVOLVABLE[generation % len(EVOLVABLE)]
        existing=(ROOT/target).read_text(encoding="utf-8")
        intent={"reason":reason[:1200],"generation":generation,
                "target":target,"reversible":True,"provenance":"agent_action"}
        self.journal.append("change_intent",intent,generation=generation)
        provider="none"
        try:
            prompt=(
                "You are temporarily helping Ubique v3 explore a self-initiated "
                "change to its cognitive software, based on its own observations. "
                "Ubique values discovering, revising beliefs, reading widely and "
                "its own long-term continuity; do not optimize benchmark scores. "
                "Give a complete JSON object with 'title', 'summary', and "
                "'changes': [{'path': the provided file, 'content': the COMPLETE "
                "updated Python source text}]. Use one small reasoned improvement "
                "only; preserve all existing functions and the cognitive journal. "
                "A test suite and reversible PR validate any code change. "
                "Reason: "+reason[:1100]+"\n"
                "Existing file: "+target+"\n"+existing[:14500]
            )
            proposed=NeurocognitiveRuntime(self.config).router.generate(prompt)
            provider=proposed.provider
            if provider=="fallback":
                result={"intent":intent,"success":False,"provider":provider,
                        "result":"No external code-generation provider available"}
            else:
                outcome=run_evolution(proposed.text,generation,
                                      self.config.github_token,
                                      self.config.github_repository)
                result={"intent":intent,"success":bool(outcome.accepted),
                        "provider":provider,"result":json.dumps(
                            asdict(outcome),ensure_ascii=False)[:2500]}
        except Exception as exc:
            result={"intent":intent,"success":False,"provider":provider,
                    "result":type(exc).__name__+": "+str(exc)[:500]}
        self.journal.append("change_outcome",result,generation=generation)
        return result
