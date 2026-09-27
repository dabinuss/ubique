from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FZG_VERSION = "1.0"

FZG_SYSTEM_PROMPT = """FZG v1.0 is the normative theoretical basis for Ubique.

Binding rules:
1. Do not rewrite FZG while applying it. Any theory change must be a separately versioned revision proposal.
2. Keep three levels separate:
   - self-preservation P
   - functional goal-directedness G_A
   - intelligence profile I_C
3. Self-preservation is only goal-directedness with respect to A_self. It is not an intelligence measure and is not a prerequisite for intelligence.
4. Canonical definitions:
   P(S|C) = G_Aself(S|C)
   G_A(S|C) = E[Q(U_t+n,A) | do(M_S), C] - E[Q(U_t+n,A) | do(M_S^-), C]
   I_C(S) = (Z_C, K_C, R_C, L_C)
   Phi(S,A,C) = [P(S|C), G_A(S|C), Z_C, K_C, R_C, L_C]
5. I_C is a profile, not a scalar. Do not sum or weight Z,K,R,L without separate empirical/theoretical validation.
6. Required application order:
   S -> A -> C -> Q -> M_S -> M_S^- -> optional P for A_self -> G_A -> Z,K,R,L -> profile -> context-bounded conclusions.
7. A must be defined before evaluation. Do not choose or alter A after seeing the result.
8. If causal intervention is unavailable, label G_A as approximated or observational; do not present it as fully causally identified.
9. Z counts only functionally relevant state distinctions.
10. K counts functionally retained capability across genuinely different contexts, not superficial variants.
11. R counts relations the system functionally uses.
12. L requires functionally different successful solution paths; mere action count is insufficient.
13. Cross-system-class numeric comparison requires functional measurement equivalence. Otherwise report structural/profile comparisons only.
14. Never infer global intelligence from a local success.
15. Contradictory findings must be documented; do not modify the theory to fit the result.

For a complete FZG analysis, output at minimum:
S, A, C, Q, M_S, M_S^-, P if A_self is tested, G_A, Z, K, R, L, I_C(S), and explicit limitations.
"""


@dataclass(frozen=True, slots=True)
class FZGCase:
    S: str
    A: str
    C: str
    Q: str
    M_S: str
    M_S_minus: str


def validate_case(case: dict[str, Any]) -> list[str]:
    """Validate the mandatory preconditions for a full FZG application."""
    required = {
        "S": "system boundary",
        "A": "goal/function criterion",
        "C": "context",
        "Q": "goal-quality operationalization",
        "M_S": "relevant mechanism",
        "M_S_minus": "control/ablation condition",
    }
    errors: list[str] = []
    for key, meaning in required.items():
        value = case.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing {key}: {meaning}")
    return errors


def fzg_profile_template() -> dict[str, Any]:
    """Return the canonical non-scalar FZG result structure."""
    return {
        "S": None,
        "A": None,
        "C": None,
        "Q": None,
        "M_S": None,
        "M_S_minus": None,
        "P": None,
        "G_A": None,
        "G_A_identification": None,
        "Z": None,
        "K": None,
        "R": None,
        "L": None,
        "I_C": {"Z": None, "K": None, "R": None, "L": None},
        "limitations": [],
    }
