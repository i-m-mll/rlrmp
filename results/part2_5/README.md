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

## Phase 4: Loss Balance Experiment — Does Adaptive Control Penalty Enable SISU→Velocity Modulation?

**Hypothesis:** The SISU signal has no effect on trajectory speed because the control cost (nn_output penalty) is too weak relative to the goal-error terms. If we adaptively rebalance them to maintain a target ratio of control cost to goal error, the network should face a genuine speed-accuracy tradeoff, giving SISU room to modulate velocity.

**What was intended:** Train 4 models with `loss_update` enabled and different `target_ratio` and `pert_std` values:
- `ratio03_pert1`: target_ratio=0.3, pert_std=1.0
- `ratio05_pert1`: target_ratio=0.5, pert_std=1.0
- `ratio03_pert10`: target_ratio=0.3, pert_std=10.0
- `ratio05_pert10`: target_ratio=0.5, pert_std=10.0

**What actually happened:** The adaptive loss update was **not enabled** during training. The `build_hps` function stores `target_ratio` from `--target-ratio` in `hps.loss_update.target_ratio` but never sets `hps.loss_update.enabled = True`. As a result:
- `ratio03_pert1` and `ratio05_pert1` are identical models (same MD5 hash), differing only in stored config metadata.
- `ratio03_pert10` and `ratio05_pert10` are identical models.
- The conditions are equivalent to `running_cost_standard` with `pert_std=1.0` and `pert_std=10.0` respectively.

**Evaluation results (run `scripts/eval_loss_balance.py`):**

| Condition           | Loss   | Ep err | Peak vel | vel@SISU=0 | vel@SISU=1 | SISU 0→1 Δvel | Lat dev (×1) | nn_output_w |
|---------------------|--------|--------|----------|------------|------------|---------------|--------------|-------------|
| running_cost_std    | 7.7219 | 0.0059 | 3.329    | 3.328      | 3.323      | -0.1%         | 0.0125       | 1.00e-05    |
| ratio03_pert1 (=ratio05_pert1) | 8.1421 | 0.0094 | 3.363 | 3.362 | 3.368 | +0.2% | 0.0169 | 1.00e-05 |
| ratio03_pert10 (=ratio05_pert10) | 7.7975 | 0.0056 | 3.306 | 3.301 | 3.314 | +0.4% | 0.0081 | 1.00e-05 |

Notes:
- Unperturbed metrics (ep_err, peak_vel) at SISU=0.5, pert_scale=0.
- SISU comparison at pert_scale=0.5 using `running_cost_standard` trial specs.
- Lateral deviation at SISU=0.5, pert_scale=1.0.

**Result: No SISU→velocity effect.** The SISU 0→1 velocity change is +0.2%–+0.4% across all conditions — consistent noise, not a real effect. This is the same null result as Phase 2.

pert_std=10.0 models show improved robustness vs pert_std=1.0 (lateral deviation 0.0081 vs 0.0169 at scale=1), confirming that stronger perturbation training helps. But the SISU velocity signature remains absent.

**The adaptive loss update still needs to be tested.** Fix required: add `--enable-loss-update` flag to `train_part2_5.py` (or auto-enable when `target_ratio` is explicitly set), retrain the 4 conditions with it actually active, and re-evaluate.

## What This Means

1. **The running cost loss works.** It's the correct loss structure for reaching tasks in the graph architecture. Other modes need debugging.

2. **PAI-ASF models learn SISU-dependent accuracy modulation** — a real robustness signature, just not the one Crevecoeur & Scott highlighted.

3. **The velocity signature is harder to produce than expected.** Neither expected-cost optimization (standard backprop), tail-risk optimization (CVaR), nor adversarial optimization (APT with various hyperparameters) generates it. Possible remaining explanations:
   - The loss structure still doesn't sufficiently reward speed (no movement-time penalty)
   - The GRU architecture may need an explicit velocity-cost tradeoff mechanism
   - The point-mass dynamics may lack the biomechanical structure that produces co-contraction/impedance-based velocity changes in humans

## Files and Data

- `figures/` — interactive plotly HTML figures, referenced inline throughout this document.
- `models/` — saved trained models, training configs, and loss histories. One subdirectory per condition, each containing `config.json`, `trained_model.eqx`, and `train_history.eqx`.
