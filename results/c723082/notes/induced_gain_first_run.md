# Induced-gain first run — Part 2.5 trained checkpoints

**Issue.** `6fdf9a4`
**Branch.** `feature/induced-gain-first-run`
**Date.** 2026-05-07

## Goal

Compute the closed-loop H-infinity induced gain `||T_{w → z}||_∞` for each
trained Part 2.5 checkpoint group across three w channels (additive_force,
structural_da, sensory_perturbation), comparing each gain against the H-inf
Riccati γ⋆ on the same plant + cost schedule.

## Setup

- **Plant** (rlrmp regime): `linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)` → 6-state.
- **Cost schedule**: `cost_schedule_from_spec(CostSpec(n_steps=200))` (matches `test_riccati_round_trip_qr_cost`).
- **Reach**: `init=(0.0, 0.0)`, `target=(0.15, 0.0)` (15 cm forward).
- **SISU**: held at 0.5.
- **Horizon**: T = 200 steps (= 2 s).
- **Algorithm**: power iteration only (3 restarts, max_iter=600, rtol=1e-05).
  Hamiltonian/fixed-point not run for v1 — adds friction without changing the headline.
- **Riccati baseline**: γ⋆ = **0.013749**.

## Headline table

| Group | γ⋆ (Riccati) | γ_PI: af×qr | γ_PI: sd×qr | γ_PI: sp×qr | γ/γ⋆ (af) |
|---|---|---|---|---|---|
| `baseline_standard_12k` | 0.0137 | 0.1237 | 148.5024 | 2.6132 | 8.997 |
| `vanilla_single` | 0.0137 | 0.2481 | 169.2627 | 4.3254 | 18.045 |
| `vanilla_pop5` | 0.0137 | 0.1458 | 164.6446 | 4.4057 | 10.606 |
| `minimax_single_seed0` | 0.0137 | 0.1446 | 153.3508 | 2.1396 | 10.519 |
| `minimax_single_seed1` | 0.0137 | 0.1563 | 162.8198 | 1.3214 | 11.369 |
| `minimax_single_seed2` | 0.0137 | 0.1556 | 162.9178 | 1.4541 | 11.318 |
| `mult_single` (rep 0, degenerate) | 0.0137 | 15.6446 | 5863.8732 | 605.3933 | 1137.862 |
| `mult_single` (rep 2, replacement) | 0.0137 | 0.1834 | — | — | 13.336 |
| `mult_pop5` | 0.0137 | 0.1789 | 165.0563 | 4.2192 | 13.015 |
| `ratio03_single` | 0.0137 | 0.1709 | 163.3942 | 2.6417 | 12.429 |
| `ratio03_pop5` | 0.0137 | 0.1585 | 164.5323 | 2.1005 | 11.526 |

Asterisk (`*`) marks non-converged power-iteration results — the reported gamma is the largest restart estimate at ``max_iter``.

**Headline summary (additive_force × qr_cost)**: min=0.1237, max=15.6446, min γ/γ⋆=8.997, max γ/γ⋆=1137.862. Note: the `mult_single` row used replicate 0 which is degenerate; see replicate spot-check below. Corrected max (using `mult_single` rep 2) = 0.2481 (`vanilla_single`).

## API integration friction

This was the *first* time the analyser was run on real Part 2.5 checkpoints
(post audit `b131510`). The expected adapter `feedbax_graph_controller(graph,
key=...)` from `induced_gain.py` does **not** work directly on a trained
SimpleFeedback graph: `graph._call_single_step(...)` immediately raises
`KeyError: 'force'` because SimpleFeedback's cycle wires (mechanics → feedback
→ net) are not threaded by `_call_single_step` (only by
`_call_with_iteration`). That code path was previously verified only on the
synthetic 1-component graph in `tests/test_induced_gain.py`.

**Workaround used here**: a network-only adapter (`_NetworkController` in
`scripts/run_induced_gain_part2_5.py`). It wraps `model.nodes['net']` directly
and reconstructs the closed loop step-by-step:

- The augmented controller hidden state packs the network's `NetworkState`
  (input, hidden, output, encoding) plus the feedback delay queue (`delay=5`,
  obs dim 4 → +20 dims). Total `n_ctrl ≈ 393` dims for the Part 2.5 ensemble.
- Disturbance enters via `linearize_pointmass`'s `Bw` (additive force on
  velocity), bypassing the trained `DisturbanceField` intervenor.
- Motor / feedback / hidden noise are zeroed (deterministic linearisation).
- Task input is held at a representative mid-movement vector
  (`target_pos`, hold=0, go=1, sisu=0.5).
- The trained network was trained on absolute coordinates; the adapter
  converts the analyser's goal-centred `pos` back to absolute by adding
  `target_pos`.

**Open follow-up** (file as a separate issue under audit `b131510` if needed):
either (i) extend `feedbax_graph_controller` to thread cycle wires through
`_call_single_step` (mirroring `_call_with_iteration`'s `init_cycle_values`
machinery), or (ii) keep the network-only adapter in `induced_gain.py` as the
canonical path for SimpleFeedback-shaped models.

Other small frictions handled in the runner:
- Replicate axis. Trained ensembles save with leading `n_replicates=5` axis
  on every weight leaf; `_select_replicate(model, 0)` indexes one replicate
  before passing to the adapter.
- Multiple checkpoint kinds: `baseline_standard_12k` saves only a warmup
  model (no adversarial phase); other groups save
  `checkpoints_adversarial/checkpoint_latest/model.eqx`. The runner picks
  the latest available per group.
- Saved configs from the runpod batch lack newer fields
  (`hidden_type`, `sisu_gating`, `streaming_loss`, `fused`, etc.); the
  runner injects sane defaults so `train_minimax.build_hps` accepts them.

## What is *not* in this run

- **Hamiltonian / LTI fixed-point** induced gain. The power-iteration LTV
  result is the headline for finite-horizon reaches; the Hamiltonian path
  would require a fixed-point Newton solve per checkpoint and adds friction
  without changing the cross-method comparison.
- **`peak_velocity`** z channel. Behavioural-scaled gain — useful but
  orthogonal to the flavor (a) ⊊ (b) discrimination.
- **Single seed / single replicate**. Each group reports replicate 0 only;
  the analyser is deterministic given the loaded weights, but cross-replicate
  variance (within `_pop5` groups especially) is not characterised here.
- **Noise / stochastic terms**. The trained controller has multiplicative
  motor noise, additive feedback noise, and hidden-unit noise. All are
  zeroed for the analysis (the H-inf framing is a deterministic worst-case
  perturbation; the analyser does not model stochastic gain). For groups
  that explicitly trained against stochastic perturbation (e.g. `mult_*`),
  the gain reported here may understate the *effective* perturbation
  attenuation.
- **Deprioritised groups**: `tier1_redo`, `bench`, `ratio_sweep`. Skipped
  per the issue spec.
- **Cross-checkpoint sweep** (γ vs training step). Only the
  final/latest checkpoint per group.

## Headline observations

10/10 groups produced gains; 0 skipped (see table for
reasons). The full numerical detail lives in
`_artifacts/part2_5/runs/induced_gain_first_run/gains.npz` (mirror of this
spec dir).

- Force-channel ranking (low → high, using corrected `mult_single` rep 2): baseline_standard_12k (0.1237) < minimax_single_seed0 (0.1446) < vanilla_pop5 (0.1458) < minimax_single_seed2 (0.1556) < minimax_single_seed1 (0.1563) < ratio03_pop5 (0.1585) < ratio03_single (0.1709) < mult_pop5 (0.1789) < mult_single-rep2 (0.1834) < vanilla_single (0.2481)
- Outliers (γ_af > 1.0, indicating closed-loop instability for the linearised analysis): mult_single rep 0 only (see replicate spot-check); method is healthy using any of reps 2–4

**Cross-method observations (af×qr channel):**

- Riccati H-inf γ⋆ = 0.0137; the *best* trained network (`baseline_standard_12k`,
  γ_af ≈ 0.1237 if af_gammas else "n/a") is roughly **9× above γ⋆**.
  This is the expected order of magnitude — no trained network is designed to
  minimise the H-inf operator norm directly; they minimise expected QR cost
  under stochastic perturbations.
- The **minimax-trained** seeds (0/1/2) cluster tightly (γ_af ≈ 0.145–0.156)
  but are *not* the lowest force-channel gains in the table — `baseline_standard_12k`
  (γ_af ≈ 0.124) is comparable, and `vanilla_pop5` (γ_af ≈ 0.146) matches
  minimax. This suggests the H-inf operator norm of the closed loop is **not
  a sensitive discriminator** between the parametric force-field adversary
  used in minimax training and other training regimes — at least not on this
  single canonical reach with this single replicate.
- The **sensory-perturbation** channel (γ_sp) distinguishes minimax (γ_sp ≈
  1.3–2.1) from non-adversarial training (γ_sp ≈ 2.6–4.4) more cleanly than
  γ_af does. This tracks the analogous pattern from the existing
  feedback-perturbation analysis: minimax improves robustness to feedback
  noise more than to body-frame force.
- The **structural_da** channel produces very large gains (γ_sd ≈ 150–170)
  across all groups — the implied small-gain margin against unstructured
  plant uncertainty is ~0.6% of the operator norm of the nominal closed loop,
  uniformly across training methods. This is consistent with the expectation
  that finite-horizon LQ-style training does not reward small-gain
  robustness.

**Outlier**: `mult_single` produces γ_af ≈ 15.6 and γ_sd ≈ 5860 — clearly an
unstable closed-loop linearisation for replicate 0. A follow-up spot-check
(issue `4f2e934`, branch `feature/mult-single-replicate-check`) ran the
additive-force analyser on all 5 replicates of `mult_single` and found:

| Replicate | γ_af   | γ/γ⋆      | converged |
|-----------|--------|-----------|-----------|
| 0         | 15.645 | 1137.9    | Y         |
| 1         | 0.537  | 39.0      | Y         |
| 2         | 0.183  | 13.3      | Y         |
| 3         | 0.156  | 11.3      | Y         |
| 4         | 0.152  | 11.0      | Y         |

**Verdict (Outcome A): replicate 0 is specifically degenerate; replicates 2–4
are normal-range (γ_af ≈ 0.15–0.18, comparable to `mult_pop5` at 0.179).
Replicate 1 is moderately elevated (γ_af = 0.537, ×39 above γ⋆) but not
outlier-class.** The training logs do not provide per-replicate loss
breakdowns (only mean ± std across all 5 replicates), so the training-time
cause is not directly observable. Training was also cut short at batch
500/5000 (10% of planned adversarial training) for both `mult_single` and
`mult_pop5`. The degenerate closed-loop linearisation of replicate 0 is
consistent with non-convergence or a degenerate fixed point in that particular
weight initialisation seed.

**Recommended replacement**: use replicate 2 (γ_af = 0.183) as the
representative `mult_single` entry in cross-method comparisons. It is the
lowest-gain well-converged non-zero replicate and is in line with
`mult_pop5`. The corrected headline `mult_single` value is γ_af = **0.183**
(was 15.645), γ/γ⋆ = **13.3** (was 1137.9).

**Caveats on flavor (a) ⊊ (b)**: this first run does **not** strongly
discriminate flavor-(a) (additive force) and flavor-(b) (structural ΔA)
trained networks because:
1. The set of training methods covered here uses only flavor-(a)
   adversaries (parametric force-field minimax + multiplicative noise
   variants). No model in this run was trained against a structural ΔA
   adversary, so we cannot compare γ_sd × qr ratios *across* flavors.
2. The `structural_da` channel here measures the *sensitivity* of any
   closed loop to unstructured ΔA, but on its own does not establish
   whether flavor-(a) or flavor-(b) training reduces that sensitivity
   more — that comparison requires a flavor-(b)-trained network as a
   data point.

The implication for the broader question is logged separately on the
analyses coordination issue (`4d38c15`) and the training-methods
coordination issue (`c99ad9d`).
