# FZG v1.0 — normative implementation basis

Ubique uses **FZG v1.0** as its binding theoretical basis.

This repository implementation preserves the directive's central separation:

- **Self-preservation**: `P(S|C) = G_Aself(S|C)`
- **Functional goal-directedness**:
  `G_A(S|C) = E[Q(U_t+n,A) | do(M_S), C] - E[Q(U_t+n,A) | do(M_S^-), C]`
- **Intelligence profile**: `I_C(S) = (Z_C, K_C, R_C, L_C)`
- **System vector**: `Phi(S,A,C) = [P, G_A, Z_C, K_C, R_C, L_C]`

## Binding application order

1. Define system `S`.
2. Define goal/function criterion `A` before evaluation.
3. Define context `C`.
4. Operationalize `Q(U,A)`.
5. Identify mechanism `M_S`.
6. Define control/ablation `M_S^-`.
7. Determine `P` only if `A_self` is being studied.
8. Determine `G_A`.
9. Only then determine `Z, K, R, L`.
10. Report a profile.
11. Draw conclusions only inside the tested `C` and tested dimensions.

## Non-negotiable interpretation rules

- High `P` does not imply high intelligence.
- Low/irrelevant `P` does not prevent a broad intelligence profile.
- Goal-directedness is not consciousness or subjective intent.
- `Z,K,R,L` are not automatically combined into one score.
- A local success is not a global intelligence claim.
- Cross-class numeric comparisons require functional measurement equivalence.
- Observational estimates of `G_A` must not be mislabeled as fully causal.
- Contradictory evidence is documented; the theory is not altered to fit results.
- Any change to a canonical formula, definition, or P/G/I relationship requires a new FZG version.

The original uploaded directive remains the authoritative source for FZG v1.0. This file is an implementation-facing canonicalization for the repository, not a revised theory.
