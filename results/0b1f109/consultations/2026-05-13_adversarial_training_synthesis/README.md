# Codex consultation on adversarial training + cs2019 induction

**Date**: 2026-05-13
**Tracking issue**: `0b1f109`
**Consultant**: Codex (GPT-5.5 via `codex exec --dangerously-bypass-approvals-and-sandbox`)
**Trigger**: User raised the concern "is fixed-adversary-at-convergence actually adversarial?" during the b399efc post-matrix discussion. Two prior archaeology passes corrected earlier Claude framings (most importantly: the `410d7ac` regulator-vs-tracker MVP was structurally degenerate, not a null result).

## Files

- `prompt.md` — verbatim prompt sent to Codex, including the consultation question framing, immediately-preceding conversational context, important context updates Claude wanted to flag, materials inventory, and the four-question structure. Plus a complexity caveat (centrality-indexed recommendations) and a task-structure context note (delayed-reach paradigm justification with savings2025 reference).
- `codex_response.md` — Codex's final response, verbatim.
- `synthesis.md` — Claude-side summary of Codex's response, as delivered to the user. (Approximately the content of issue `0b1f109` body, with extra structure.)

## Key conclusions (see issue `0b1f109` for details)

- The original "fixed adversary = not adversarial" worry is real but mis-attributed: the right concern is the combination of nonconvex RNN + weakly-verified PGD + no held-out worst-case audit.
- The current `LinearDynamicsAdversary` is a different game from cs2019's full-state additive ε H∞.
- Multi-signature framing is empirically supported but not centrally indexed: feedback-gain modulation is broad and easy; speed inflation is narrow.
- Single must-do next experiment: implement full-state ε adversary matching cs2019 Eq 13 + cs2019's (t/N)^6 cost schedule + PGD step sweep, with linear regulator as Riccati round-trip gate before any GRU arm.
