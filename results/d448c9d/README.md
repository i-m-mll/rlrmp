# d448c9d — linear regulator vs tracker decoupling acid test

Stage 2 of phase umbrella `f695729` (regulator-coupling and biomechanical decoupling — staged programme). Tests whether the regulator-vs-tracker parameterisation explains why our trained GRU controllers exhibit robustness without the analytically-predicted +Δv peak-velocity-inflation signature.

The MVP child `410d7ac` ran the first laptop-CPU version (four matched linear models: regulator/tracker × baseline/adversarial) and produced a structurally degenerate result — both architectures gave Δv ≈ +78% because the MVP's "tracker" had a trivial constant x_nom, making it an affine LTV regulator-to-target rather than a real trajectory tracker. See `410d7ac` for the MVP write-up and retraction trail.

This directory holds:
- `consultations/` — external-agent reviews of the framing and path forward (e.g. `2026-05-11_decoupling_framing/` for Gemini 3.1 Pro + GPT-5.5 via codex).
- `notes/` — narrative and analysis writeups produced from the consultations and any follow-up work scoped here.

Note: this is the parent / coordination-style hash dir for the decoupling acid test as a whole. Per-tier work (Tier 1 GRU affine decomposition, Tier 2 fixed-direction matched-objective sweep, Tier 3 GRU under matched objective) lives under its own tracking issue and `results/<tier-hash>/` once filed.
