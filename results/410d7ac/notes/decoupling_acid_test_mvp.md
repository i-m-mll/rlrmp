# Decoupling acid test — MVP first-signal results

Tracking issue: `410d7ac` · parent `d448c9d` · phase umbrella `f695729` ·
training-methods coord `c99ad9d`.

## TL;DR

| Architecture                | Δv@0.5      | Δv@1.0      | Δv@1.5      |
|----------------------------|-------------|-------------|-------------|
| `linear_regulator`         | +0.010 ± 0.009 | +0.035 ± 0.015 | +0.060 ± 0.019 |
| `linear_tracker`           | +0.011 ± 0.009 | +0.036 ± 0.015 | +0.061 ± 0.019 |
| `gru_baseline_lit_post_nojerk` (f47abb1) | +0.002 ± 0.001 | +0.004 ± 0.002 | +0.005 ± 0.002 |

Δv is the relative inflation of peak forward velocity under a `LinearDynamics`
plant disturbance (scale = SISU level), averaged over 5 replicates × 8 reach
directions.

**The discriminator did NOT split cleanly.** The tracker behaved identically
to the regulator in this MVP. Mechanism: the optimiser never used the
independent `u_ff(t)` channel — `‖u_ff‖_2` stayed at ~0.01 (max 0.02) while
`‖K_t · e_t‖` was on the order of 1.0. The tracker effectively *was* the
regulator. That means the algebraic identity flagged in d448c9d comment
d444498 (tracker collapses to regulator when `u_ff = -K · x_nom`) holds here
as a near-equality empirically, even with `u_ff` and `K` independently
parameterised — not because the math forces it, but because the optimiser
preferred a regulator-only solution given the training signal and budget.

Both linear architectures show Δv > 0 (regulator-like signature). The GRU
baseline shows Δv ≈ 0.

## Caveats / what this MVP can and cannot conclude

- **Pre-merger interpretation.** The tracker is *capable* of decoupling — its
  parameter space is a strict superset of the regulator's — but at 1000 batches
  on laptop CPU the optimiser did not find the decoupled solution. The MVP
  result is consistent with both "decoupling is hard to learn" and "decoupling
  is fundamentally not preferred by the loss landscape". The MVP cannot
  distinguish.
- **Peak velocities are very different across architectures.** GRU baseline
  peaks at ~0.97 m/s; linear controllers peak at ~0.16 m/s. The linear models
  trained for 1000 batches (laptop budget) vs 12000 for the GRU. Δv is a
  *relative* quantity, so the comparison across architectures is still
  meaningful, but the linear controllers are clearly under-trained.
- **`x_nom(t) ≡ 0` design choice.** The tracker uses `x_nom = 0` in the
  target-relative frame, so the decomposition is `u = u_ff(t) − K_t · e_t`
  where `e_t = (pos − target, vel)`. A different `x_nom` (e.g. a
  constant-velocity straight-line trajectory) would change the gradient
  landscape and might encourage a non-zero `u_ff`. This is a candidate
  follow-up.
- **No adversarial training.** `--n-adversary-batches 0` so the warmup-only
  models are not robustified. The Δv signature here is the *standard-trained*
  signature — the discriminator framing from d448c9d. A trained adversary
  could push the regulator toward Δv > 0 more sharply or force the tracker
  toward decoupling; the MVP does not test this.

## Discriminator outcome (per d448c9d framing)

- Gemini's claim "linear regulator inflates velocity": **supported** in this
  MVP (regulator Δv > 0, monotone in disturbance scale).
- Decoupling-via-parameterisation hypothesis: **NOT supported** in this MVP —
  the tracker did not learn to decouple. But this is a learning-dynamics
  null, not a theoretical null. The math still admits decoupling; the
  optimiser just did not find it under this training protocol.

## Follow-ups motivated

1. **Longer training + better initialisation.** Re-run at 5000–10000 batches
   to give the optimiser room to find a decoupled solution. Initialise `u_ff`
   from a precomputed open-loop bang-bang reach (warm start the feedforward
   channel).
2. **Non-trivial `x_nom`.** Add a per-trial nominal velocity profile and have
   the tracker use it; this raises the cost of `u_ff = 0` because then the
   regulator has to do the tracking work alone.
3. **Adversarial phase.** Couple the existing `LinearDynamicsAdversary` to
   each architecture and look at whether minimax training rebuilds the
   discriminator (regulator stays Δv > 0, tracker drops to Δv ≈ 0).
4. **Affine recurrent + GRU comparators** (variants 3 and 4 in `d448c9d`)
   to span the full architecture menu.

## Files

- Models: `_artifacts/410d7ac/runs/{linear_regulator,linear_tracker}/warmup_model.eqx`
- Per-run specs: `results/410d7ac/runs/{linear_regulator,linear_tracker}.json`
- Δv summary: `results/410d7ac/notes/delta_v_summary.json`
- Figure: `results/410d7ac/figures/delta_v_signature/`
- Training logs: `_artifacts/410d7ac/logs/{linear_regulator,linear_tracker}.log`
- Implementation: `src/rlrmp/networks/linear_controllers.py` (+ touchpoints
  in `src/rlrmp/models.py`, `src/rlrmp/modules/training/part2.py`,
  `scripts/train_minimax.py`).

## Cross-refs

- Comment for tracking issue `410d7ac` (this file is the canonical writeup).
- Comment for training-methods coord `c99ad9d` if the MVP motivates a tier
  shift on the regulator-vs-tracker question.
- Parent `d448c9d`: the MVP closes one of the four architecture variants for
  the laptop-CPU subset; variants 3 (affine recurrent) and 4 (GRU control)
  remain.
