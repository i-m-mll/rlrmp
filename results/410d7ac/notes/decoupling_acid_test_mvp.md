# Decoupling acid test — corrected MVP results

Tracking issue: `410d7ac` · parent `d448c9d` · phase umbrella `f695729` ·
training-methods coord `c99ad9d`.

> **No quantitative correction needed after the go-cue alignment fix
> (Bug: 06f7faf).** The primary `Δv` metric here is computed from peak
> forward velocity, which uses per-trial `after_go` masking before the
> `max` reduction — it never collapsed the trial axis in absolute time.
> The fix re-rendered the side-by-side velocity-profile figure with the
> x-axis recentred on the go cue (t=0), but the Δv numbers reported in
> this note are unchanged. The figure spec / render under
> `results/410d7ac/figures/delta_v_signature/` was regenerated post-fix
> for visual accuracy only.

This is the **corrected** MVP. The prior MVP (commit `20ae797`, results
reported on `410d7ac` comment 3) trained warmup-only models and measured
test-time perturbation response, which is NOT Δv. The retraction is documented
on `410d7ac` comment 4.

## What Δv actually is

Δv is the **peak forward velocity inflation between an adversarially-trained
model and the warmup-only baseline of the SAME architecture**, signed and
projected on the reach axis:

    Δv_arch = (peak_v(arch_adversarial) - peak_v(arch_baseline)) / peak_v(arch_baseline)

It is a *training-method comparison*. Both members of each architecture pair
share their seed and warmup schedule; the only difference is the adversarial
phase. The d448c9d discriminator prediction is `Δv_regulator > 0` (adversarial
training inflates regulator peak velocity) and `Δv_tracker ≈ 0` (the tracker's
feedforward channel absorbs the disturbance without changing peak velocity).

## TL;DR

Four models trained (seed=42, warmup=1000 batches, adversarial=500 batches
against `LinearDynamicsAdversary` ΔA·x perturbation, eta_max=0.1):

| Architecture | Δv @ pert=0 | Δv @ pert=1 |
|---|---|---|
| `linear_regulator` | **+0.775 ± 0.034** (SEM, n=5) | +0.718 ± 0.037 |
| `linear_tracker`   | **+0.776 ± 0.034** (SEM, n=5) | +0.717 ± 0.036 |

Peak forward velocity (m/s, mean across 5 replicates × 8 reach directions):

| Run | peak_v @ pert=0 | peak_v @ pert=1 |
|---|---|---|
| `linear_regulator__baseline`    | 0.158 ± 0.006 | 0.164 ± 0.006 |
| `linear_regulator__adversarial` | 0.280 ± 0.006 | 0.281 ± 0.006 |
| `linear_tracker__baseline`      | 0.158 ± 0.005 | 0.163 ± 0.006 |
| `linear_tracker__adversarial`   | 0.280 ± 0.007 | 0.280 ± 0.006 |

The Frobenius norm of the LinearDynamicsAdversary's ΔA matrix saturated the
0.1 projection ceiling for both adversarial runs (`mean=0.1 ± 6e-9`), so the
threat budget was fully exercised.

## Discriminator outcome — second informative null

**Did not split cleanly.** Both architectures show a large +78% Δv signature
at the headline evaluation condition (pert_scale=0). Adversarial training
substantially inflated peak velocity for both regulator and tracker. The
discriminator framing from `d448c9d` is NOT supported by this corrected MVP.

## u_ff utilization — quantitatively higher, qualitatively still inactive

|u_ff| stats from the tracker controllers:

| Run | |u_ff|_max | |u_ff|_L2 mean | |K|_F mean | peak_v |
|---|---|---|---|---|
| `linear_tracker__baseline`    | 0.021 | 0.010 | 5.42 | 0.158 |
| `linear_tracker__adversarial` | 0.052 | 0.040 | 7.40 | 0.280 |

Adversarial training increased `|u_ff|` by ~4× (max) and ~4× (L2 mean), so
the optimiser *did* push more signal through the feedforward channel than the
warmup-only MVP did (the prior MVP reported |u_ff|_L2 ≈ 0.01 for the tracker).
But the magnitude is still small compared to the feedback contribution:
typical |K · e| ~ |K|·|e| with K_max ~ 4.8 and e in the order of 0.1 m gives
|K · e| ~ 0.5 — an order of magnitude larger than max |u_ff| = 0.05. The
tracker is still operating in the regime where K dominates u_ff.

## Mechanism (working hypothesis)

The tracker's parameter space strictly contains the regulator's, so the
inflation observed in the regulator's Δv is also achievable by the tracker —
either via larger K or via larger u_ff. Both are loss-equivalent at first
order, and the optimiser is choosing the K-heavy solution (the same one the
regulator is forced into). The structural prediction "u_ff can absorb the
disturbance without changing peak velocity" does NOT manifest at this
training budget because the loss landscape has many K-only minima that
adversarial gradient ascent finds first.

Two possible interpretations:

1. **The decoupling-via-parameterisation hypothesis is wrong.** The regulator
   and tracker face the same loss surface near zero u_ff; adversarial training
   just inflates K for both. The hypothesised "tracker absorbs the threat
   into u_ff" requires a structural inductive bias the optimiser does not
   have.
2. **The decoupling is hard to discover under this protocol.** 500
   adversarial batches with controller_lr=5e-3 may be insufficient. A
   different protocol (longer adversarial phase, u_ff warm start from a
   precomputed bang-bang, non-trivial x_nom, larger eta_max so the disturbance
   threat is large enough that staying-on-K solutions are sub-optimal) could
   discover decoupling.

The MVP cannot distinguish these two interpretations.

## Caveats

- **K dominates u_ff throughout.** As described above, even adversarial
  training did not promote u_ff into the same magnitude regime as K · e.
  Initialising u_ff at zero may bias the optimiser toward K-only solutions
  early in training.
- **x_nom ≡ 0 in target-relative coords** (same as the prior MVP) — the
  tracker reduces to "regulator + zero-mean u_ff" by construction. With a
  non-trivial nominal trajectory u_ff would have a non-zero target and
  optimisation might be pushed off the regulator manifold.
- **Validation loss did not strictly improve under adversarial training**
  for either architecture, but the controllers are not catastrophically
  worse either: ctrl_loss settled near 1.30 (adversarial) vs warmup final
  ~1.55. Peak velocity *did* increase substantially, which is the standard
  signature of robust control even when nominal performance is not improved.
- **Small adversary budget (eta_max=0.1).** A larger threat region might
  force structural differences between regulator and tracker. Phase 2 of
  the parent d448c9d (full-scope acid test) could sweep eta_max.

## Discriminator outcome (per d448c9d framing)

- "Linear regulator inflates velocity under adversarial training":
  **strongly supported** (Δv = +0.78, an order of magnitude larger than the
  GRU baseline's typical Δv ~ 0.01-0.05). This is a robust positive signal
  that the regulator-coupling claim from `d448c9d` carries over to numerical
  models, not just analytical Riccati.
- "Decoupling-via-parameterisation": **not supported** in this MVP. The
  tracker behaves identically to the regulator under the same training
  protocol. This is a second informative null; the first was that warmup-only
  training already collapsed u_ff to zero (prior MVP comment 3).

## Follow-ups motivated

1. **u_ff warm start.** Initialise the tracker's u_ff to a precomputed
   bang-bang or minimum-jerk feedforward profile, then run the same
   adversarial training. Test whether the optimiser preserves the
   decoupled structure given a good initialisation.
2. **Non-trivial x_nom.** Add a per-trial nominal velocity profile and have
   the tracker use it; this raises the cost of `u_ff = 0` because the
   regulator has to do the tracking work alone.
3. **Larger adversary budget.** Sweep `linear_dynamics_eta_max` (0.1 → 0.2
   → 0.5) to see whether the discriminator opens at a larger threat scale.
4. **Longer adversarial phase.** 500 batches → 5000-10000 batches with
   `--resume` so it can be paused/continued. The current 500-batch budget is
   tight for laptop CPU verification but may be too short to discover
   structural differences.
5. **Full 4-architecture sweep** from parent `d448c9d` — affine recurrent
   + GRU comparators under the same adversarial protocol.
6. **Compare to LQR/H-inf analytical bound.** Use the existing
   `compute_velocity_inflation` machinery on the same plant + reach to
   verify the expected analytical Δv for these gains. If the analytical Δv
   at eta_max=0.1 is also ~+0.8, the linear-MVP is recovering the right
   number for the wrong reason; if it's much smaller, the linear MVP is
   inflating peak vel by some other mechanism (e.g. control-saturation
   chatter) that needs investigation.

## Files

- Models: `_artifacts/410d7ac/runs/{linear_regulator,linear_tracker}__{baseline,adversarial}/`
- Per-run specs: `results/410d7ac/runs/<variant>/run.json` (4 variants)
- Δv summary: `results/410d7ac/notes/delta_v_summary.json`
- Figure: `results/410d7ac/figures/delta_v_signature/`
- Training logs: `_artifacts/410d7ac/runs/<variant>/train.log`
- Implementation: `src/rlrmp/networks/linear_controllers.py`,
  `scripts/train_minimax.py` (`_get_trainable`, `_trainable_where`,
  `_make_where_train` linear-controller branches),
  `scripts/analyse_linear_decoupling_mvp.py` (this file's analysis).

## Resumability check (verified)

`scripts/train_minimax.py` supports full mid-training resumption for the
adversarial phase via `--resume`:

- `warmup_model.eqx` is saved after phase 1; on resume with `--resume`, the
  script loads it and skips phase 1 entirely.
- `checkpoints_adversarial/checkpoint_latest/` is overwritten every
  `--checkpoint-every` batches (we used 100). Each checkpoint contains
  `model.eqx`, `adversary_<i>.eqx`, `ctrl_opt_state.eqx`,
  `adv_opt_state_<i>.eqx`, and `meta.json` with `batch_idx`, `n_adversaries`,
  `adv_losses`, `ctrl_losses`, and `adv_indices`.
- On `--resume`, the script loads the checkpoint and continues from
  `last_completed_batch + 1`.

Caveat: the adversarial sampling RNG (`key_adv`) is not explicitly
checkpointed; it is re-derived from `--seed` and advanced via repeated
`jr.split` to the resumed batch index. This produces a deterministic
continuation but NOT bitwise identical to an uninterrupted run.

To extend any of these 500-batch runs to e.g. 5000 batches later, re-invoke
`train_minimax.py` with the same flags plus `--resume` and
`--n-adversary-batches 5000`; phase 1 is skipped, phase 2 continues from
batch 500.

For baselines (n_adversary_batches=0), resumption mid-warmup is NOT
supported — only re-runnable with a larger `--n-warmup-batches`. The baseline
runs in this MVP are short enough that this is not a problem.

## Cross-refs

- Tracking issue `410d7ac` (this file is the canonical writeup; a comment
  summary on the issue links here).
- Coordination `c99ad9d` (training methods) — comment notes the cross-cutting
  finding that adversarial training with `LinearDynamicsAdversary` does NOT
  recruit a freely-parameterised feedforward channel under default
  initialisation. Decoupling-via-parameterisation as a *general* feature of
  the tracker structure is not supported.
- Parent `d448c9d`: the corrected MVP closes the first round of the 4-arch
  acid test on the laptop-CPU subset (variants 1+2). Variants 3 (affine
  recurrent) and 4 (GRU control) remain. The phase-2 follow-ups above are
  the natural next questions.
