# Part 2.5 Experiment: Does SISU Modulate Peak Velocity?

## Background

Crevecoeur & Scott (2019) showed that humans increase peak reaching velocity when facing unpredictable perturbations — a signature of H-infinity robust control. Our PAI-ASF models receive a SISU input (0 = no perturbation expected, 1 = full perturbation expected). The question: does training with perturbation uncertainty produce the same velocity increase?

## Phase 1: Which Loss Function Works?

We needed to find a loss function that trains stably with the new feedbax graph architecture before we could test anything else. Four loss modes were compared, all using standard backprop with PAI-ASF and 5 replicates.

**Running cost** (constant position error penalty from the go cue through trial end) was the only loss mode that converged. Softmin and the default structured ramps both diverged catastrophically — likely due to incompatibilities with the graph architecture's new intervenor handling. The combined mode (weak running cost + strong softmin) trained but produced mediocre results.

| Loss mode | Description | Loss (initial→final) | Endpoint error | Peak speed | Status |
|---|---|---|---|---|---|
| running_cost | Constant penalty from go cue | 32.3 → 2.9 | 0.007 | 3.33 | **Converged** |
| softmin | Goal-hit window (softmin) | 5.0 → 843 | — | — | Diverged |
| default | Structured mid/late ramps | 14.9 → 885 | — | — | Diverged |
| combined | Weak running + strong softmin | 10.7 → 5.0 | 0.152 | 2.06 | Mediocre |

See `figures/fig_loss_curves.html` for training curves of all four modes.

Models saved in `models/running_cost_standard/`, `models/softmin_standard/`, `models/default_standard/`, `models/combined_standard/`.

## Phase 2: Does SISU Modulate Velocity?

With running_cost as the loss, we tested three training objectives and three control cost levels. The key measurement: evaluate the trained model at SISU=0 vs SISU=1 with the same perturbation and compare peak velocities.

### The answer: no.

Across every converged condition, SISU produces **no increase in peak velocity** (changes range from -1.2% to +0.1%, all within noise):

| Condition | Loss (init→final) | Ep error | Peak speed | SISU 0→1 velocity | Notes |
|---|---|---|---|---|---|
| Standard backprop | 32→2.9 | 0.007 | 3.33 | -0.3% | Best overall convergence |
| CVaR 10% | 32→16.6 | 0.109 | 3.18 | -0.9% | Harder training, less accurate |
| APT (2k batches) | 32→3.0 | 0.006 | 3.27 | +0.1% | Quick TPU run |
| APT (10k batches) | 32→9.9 | 0.078 | 3.23 | -1.2% | Worse convergence than shorter runs |
| APT (lr=0.001) | 32→2.9 | 0.005 | 3.31 | -0.3% | Conservative adversary |
| APT (lr=0.1) | 32→2.9 | 0.006 | 3.31 | -0.1% | Aggressive adversary |
| APT (5 inner steps) | 32→2.9 | 0.004 | 3.30 | -0.2% | More inner optimization |
| APT (pert_std=2) | 32→2.9 | 0.005 | 3.32 | -0.1% | Stronger perturbations |
| nn_output=1e-6 | 32→5.4 | 0.070 | 3.91 | -0.7% | Lower control cost → faster |
| nn_output=1e-4 | 32→714 | 0.780 | 2.63 | — | Diverged |

See `figures/fig_peak_velocity_by_sisu.html` and `figures/fig_endpoint_error_by_sisu.html`.

### But SISU does modulate accuracy.

A decoupled test — fixed perturbation amplitude (scale=0.5), varying only the SISU input signal — reveals that the network IS using SISU:

| SISU input | Peak velocity | Endpoint error | Lateral deviation |
|---|---|---|---|
| 0.00 | 3.330 | 0.0070 | 0.0129 |
| 0.25 | 3.329 | 0.0065 | 0.0123 |
| 0.50 | 3.327 | **0.0062** | **0.0122** |
| 0.75 | 3.321 | 0.0069 | 0.0124 |
| 1.00 | 3.321 | 0.0069 | 0.0132 |

Endpoint error improves by ~11% at moderate SISU (0.0070 → 0.0062). Peak velocity is flat. This is feedback gain modulation: the network increases its corrective gains when told to expect perturbations, producing better accuracy without changing its movement speed.

This is the LQG separation principle in action: expected-cost optimization (and even CVaR/APT approximations to worst-case) changes feedback gains but not the trajectory shape. The velocity signature specifically requires the trajectory itself to change — and none of our training methods produced that.

## Phase 3: Is the Training Actually Producing Robustness?

The SISU-velocity null result raises a question: are the perturbations during training sufficient to induce robustness at all? We compared models trained with perturbations (pert_std=1) against baselines trained without (pert_std=0), evaluating all models under the same gust perturbations (from the trained model's task).

**Important methodological note:** The baseline model's own task generates zero-amplitude gusts (pert_std=0). To compare fairly, all models must be evaluated using the *trained* model's task, which generates non-zero gusts. Otherwise the baseline appears to be unaffected by any perturbation scale (scaling zero is still zero).

### Training with perturbations produces clear robustness.

Max lateral deviation under fixed gust perturbation at SISU=0.5:

| Model | scale=0.5 | scale=1.0 | scale=2.0 | scale=5.0 |
|---|---|---|---|---|
| Baseline (pert_std=0) | 0.0165 | 0.0175 | 0.0200 | 0.0291 |
| Standard (pert_std=1) | 0.0122 | 0.0127 | 0.0145 | 0.0219 |
| APT lr=0.001 (pert_std=1) | **0.0072** | **0.0079** | **0.0103** | **0.0192** |

Standard training reduces lateral deviation by ~26% vs baseline. APT reduces it by ~55% at moderate perturbations and ~34% at strong perturbations. The perturbation training IS working — APT in particular produces substantially more robust controllers.

However, even the baseline handles these perturbations fairly well (max deviation 2.9% of reach at scale=5). The perturbation amplitudes may still be too weak to fully separate robust from non-robust behavior. A perturbation strength sweep (pert_std=2, 5, 10, 20 during training) is running on TPU to test this.

### Perturbation size sweep

Models with pert_std = 2, 5, 10, 20 were trained (see `models/pert_std_*/`). Results: robustness improves with higher pert_std and the models still converge well through pert_std=10. At pert_std=20 there may be degradation; eval pending.

## Phase 4: Does Adaptive Control Cost Enable Velocity Modulation?

### Background

With the running_cost loss, position error dominates and the model operates at ceiling speed — SISU cannot push it higher. The adaptive control cost (`loss_update`) dynamically increases the `nn_output` penalty weight to balance control cost against goal error. This reduces peak speed and potentially creates room for SISU to modulate velocity.

### Setup

4 conditions: target_ratio ∈ {0.3, 0.5} × pert_std ∈ {1, 10}, all trained with `--enable-loss-update`, 10k batches, running_cost loss. The adaptive update runs every 100 iterations with alpha=0.005.

### Results

| Condition | Val loss | Ep err | Peak vel | SISU 0→1 Δvel | Lat dev (×1) | nn_output wt |
|---|---|---|---|---|---|---|
| baseline (no update) | 7.72 | 0.0059 | 3.329 | -0.1% | 0.0125 | 1e-5 |
| ratio=0.3, pert_std=1 | 10.58 | 0.0051 | 2.424 | +0.2% | 0.0063 | 3.7e-5 |
| ratio=0.5, pert_std=1 | 9.93 | 0.0045 | 2.581 | +0.2% | 0.0069 | 2.9e-5 |
| ratio=0.3, pert_std=10 | 10.66 | 0.0047 | 2.435 | +0.6% | 0.0066 | 3.7e-5 |
| ratio=0.5, pert_std=10 | 10.01 | 0.0048 | 2.586 | +0.4% | 0.0065 | 2.9e-5 |

### Interpretation

- The adaptive control cost successfully reduced peak velocity from 3.33 to 2.4–2.6 and improved lateral robustness (~50% reduction in deviation).
- SISU velocity direction flipped from slightly negative (-0.1%) to slightly positive (+0.2% to +0.6%). This is the correct direction but the magnitude is negligible.
- Higher pert_std produced marginally larger effects (+0.6% at pert_std=10 vs +0.2% at pert_std=1), consistent with stronger perturbation training creating clearer SISU expectations.
- The nn_output weight converged to 3–4× its initial value (3e-5 to 4e-5 from 1e-5) — the adaptive mechanism works but the weight increase is modest.

Models saved in `models/ratio03_pert1_v4/`, `models/ratio05_pert1_v4/`, `models/ratio03_pert10_v4/`, `models/ratio05_pert10_v4/`.

## What This Means

1. **The running cost loss works.** It's the correct loss structure for reaching tasks in the graph architecture. Other modes need debugging.

2. **PAI-ASF models learn SISU-dependent accuracy modulation** — a real robustness signature, just not the one Crevecoeur & Scott highlighted.

3. **The velocity signature is harder to produce than expected.** Neither expected-cost optimization (standard backprop), tail-risk optimization (CVaR), nor adversarial optimization (APT with various hyperparameters) generates it. Possible remaining explanations:
   - The loss structure still doesn't sufficiently reward speed (no movement-time penalty)
   - The GRU architecture may need an explicit velocity-cost tradeoff mechanism
   - The point-mass dynamics may lack the biomechanical structure that produces co-contraction/impedance-based velocity changes in humans

4. **Adaptive control cost reduces peak speed and improves robustness but does not produce a meaningful SISU → velocity signature.** The ~3–4× weight increase is insufficient to create exploitable speed headroom. A much larger control penalty (or different loss structure such as an explicit movement-time penalty) may be needed.

## Files and Data

- `figures/` — interactive plotly HTML figures, referenced inline throughout this document.
- `models/` — saved trained models, training configs, and loss histories. One subdirectory per condition, each containing `config.json`, `trained_model.eqx`, and `train_history.eqx`.
