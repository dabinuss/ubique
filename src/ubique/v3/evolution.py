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
from ..evolution import _safe_path
from .storage import Journal

EVOLVABLE = (
    "src/ubique/v3/curiosity.py",
    "src/ubique/v3/planner.py",
    "src/ubique/v3/workspace.py",
    "src/ubique/v3/semantics.py",
)

def evolving_modules() -> list[str]:
    """Every ordinary cognitive Python module may evolve, not just four."""

    candidates=list(EVOLVABLE)
    for path in sorted((ROOT/"src"/"ubique").rglob("*.py")):
        relative=path.relative_to(ROOT).as_posix()
        try:
            _safe_path(relative)
        except ValueError:
            continue
        if relative not in candidates:
            candidates.append(relative)
    return [p for p in candidates if (ROOT/p).is_file()]


class Evolution:
    def __init__(self, journal: Journal, config: Config):
        self.journal=journal
        self.config=config

    def attempt(self, reason: str, *, generation: int, target: str = ""):
        candidates=evolving_modules()
        if not candidates:
            return {"success":False,"provider":"none","result":"No evolvable module available"}
        if target not in candidates:
            # Select the part of the living cognition path relevant to the
            # observed problem; retain the entire ordinary source surface.
            linked=(
                ("library", "src/ubique/v3/library.py"),
                ("reading", "src/ubique/v3/actions.py"),
                ("source", "src/ubique/v3/semantics.py"),
                ("evidence", "src/ubique/v3/evidence.py"),
                ("comparison", "src/ubique/v3/semantics.py"),
                ("question", "src/ubique/v3/inquiry.py"),
                ("experiment", "src/ubique/v3/experimentation.py"),
                ("provider", "src/ubique/v3/providers.py"),
                ("memory", "src/ubique/v3/storage.py"),
            )
            target=next((path for word,path in linked
                         if word in reason.lower() and path in candidates), "")
            if not target:
                active=[p for p in candidates if p.startswith("src/ubique/v3/")]
                pool=active or candidates
                target=pool[generation % len(pool)]
        existing=(ROOT/target).read_text(encoding="utf-8")
        # Concrete experiences, not a fixed scalar score, inform proposals.
        history=[e.get("data",{}) for e in self.journal.recent("action_outcome",limit=24)]
        failures=[{"kind":e.get("kind"),"effect":str(e.get("effect",""))[:280],
                   "reason":str(e.get("reason",""))[:180]}
                  for e in history if e.get("success") is False][-6:]
        changes=[{"success":e.get("data",{}).get("success"),
                  "result":str(e.get("data",{}).get("result",""))[:250]}
                 for e in self.journal.recent("change_outcome",limit=5)]
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
                "updated Python source text}]. Suggest one or several coherent source changes when justified "
                "without erasing memories or journal continuity. "
                "A test suite and reversible PR validate any code change. "
                "Describe the observable change expected in the running v3 "
                "cycle, and update its active call path when needed. Prefer "
                "substantive behavior over unused helpers or cosmetic edits. "
                "Keep exploratory beliefs revisable. "
                "Prefer repairing a demonstrated failure or enabling a new "
                "cognitive behavior rather than merely changing numerical "
                "weights, unless an observed effect justifies those weights. "
                "Reason: "+reason[:1100]+"\n"
                "Recent failed actions: "+json.dumps(failures,ensure_ascii=False)[:1800]+"\n"
                "Recent change outcomes: "+json.dumps(changes,ensure_ascii=False)[:1400]+"\n"
                "Existing file: "+target+"\n"+existing[:13000]
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
