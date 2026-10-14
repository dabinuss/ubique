"""Conservative, auditable checks between a proposed belief and actual passages.

Lexical checks are *not* semantic entailment. They can reject obvious unrelated
or inverted claims, but do not certify truth. Disagreements stay unresolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from .hippocampus import terms

DENIALS = {"not", "never", "no", "without", "none", "cannot", "kein", "keine",
           "nicht", "niemals", "weder", "false", "incorrect", "impossible"}
SOURCE_STATUSES = {"source_passage", "observed", "observed_action_outcome"}
SYSTEM_MARKERS = ("runtime", "heartbeat", "provider", "github", "groq", "gemini",
                  "issue:", "issue closure", "smoke test", "preflight",
                  "workflow", "pytest", "autonomous cycle",
                  "api key", "token", "rate limit", "deployment", "telemetry")

def informative_tokens(text: str) -> set[str]:
    noisy = {"source", "passage", "study", "people", "human", "different", "distinct",
             "evidence", "suggests", "claim", "observed", "proposed", "about",
             "things", "might", "possibly", "tentatively", "should", "would"}
    return {w for w in terms(text, 70) if len(w) >= 5 and w not in noisy}

def is_system_topic(text: str) -> bool:
    low = str(text).casefold().strip()
    old_diagnostic_labels = {"state","status","continuous","newly","constraints",
                             "probe","verify","deactivated","already_requested",
                             "heartbeat","heartbeats"}
    return low in old_diagnostic_labels or any(marker in low for marker in SYSTEM_MARKERS)

@dataclass(frozen=True)
class EvidenceAssessment:
    verdict: str  # unrelated / contested / plausible / directly_supported
    overlap: float
    matches: int
    cited: int
    independent_sources: int
    explicit_negation_conflict: bool
    limitations: str

    def as_dict(self):
        return asdict(self)

def assess(statement: str, cited_episodes: list[dict]) -> EvidenceAssessment:
    quoted = [e for e in cited_episodes if e.get("epistemic_status") in SOURCE_STATUSES
              and str(e.get("text","")).strip()]
    if not quoted:
        return EvidenceAssessment("unrelated",0,0,0,0,False,"No retrievable source passage")
    claim = informative_tokens(statement)
    if len(claim) < 2:
        return EvidenceAssessment("unrelated",0,0,len(quoted),0,False,
                                  "Insufficient specific content in interpretation")
    best = 0.
    common = 0
    contradictions = 0
    independent = set()
    for episode in quoted:
        passage = str(episode.get("text",""))[:4000]
        passage_tokens = informative_tokens(passage)
        # Use an actual nearby claim span rather than combining unrelated
        # words and negations from a whole multi-paragraph chapter.
        spans = re.split(r"(?<=[.!?])\s+", passage)
        closest = max(spans, key=lambda x:len(claim & informative_tokens(x)),
                      default=passage)
        local_hits = len(claim & informative_tokens(closest))
        hits = min(len(claim & passage_tokens),max(local_hits,2*local_hits))
        coverage = hits/max(1,len(claim))
        best = max(best,coverage)
        common = max(common,hits)
        independent.add(str(episode.get("source") or episode.get("id")))
        assertion_neg = bool(set(re.findall(r"\b[\w]+\b",statement.casefold())) & DENIALS)
        local_neg = bool(set(re.findall(r"\b[\w]+\b",closest.casefold())) & DENIALS)
        if local_hits >= 3 and (local_hits/max(1,len(claim))) >= .65 and assertion_neg != local_neg:
            contradictions += 1
    if contradictions:
        verdict = "contested"
    elif best >= .80 and common >= 3:
        verdict = "directly_supported"  # lexical correspondence only, NOT verified entailment
    elif best >= .45 and common >= 2:
        verdict = "plausible"
    else:
        verdict = "unrelated"
    return EvidenceAssessment(verdict,round(best,3),common,len(quoted),len(independent),
             contradictions>0,"Lexical screening only; independent truth/entailment not established")
