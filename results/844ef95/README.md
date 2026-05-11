# Part 2.5 Experiment: Does SISU Modulate Peak Velocity? (legacy archive)

> **Archive note (f485c26 reorg).** This dir is the canonical hash-dir target for
> the legacy Part 2.5 narrative + pre-migration training-run dirs under `models/`,
> `running_cost_nn1e6/`, the top-level `config.json`, and the four
> `centerout_*_pert1/` top-level dirs. These predate the `run.json` convention and
> carry only legacy `config.json` payloads — no `run.json` migration was performed
> per the f485c26 final decisions (legacy archive policy). The original narrative
> body follows.

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

## Phase 5: Center-Out Translation-Invariant Task

### Background

The original task sampled random (start, target) pairs in a [-1,1]^2 workspace. The network had to learn translation invariance from data. A new center-out task (Bug: be3804f) enforces translation invariance by starting all reaches at origin with constant length 0.5 in random directions. This simplifies the task, reduces input dimensionality waste, and matches how the point-mass dynamics (F=ma) are translation-invariant.

### Key finding: pert_std=0 now converges

The pert_std=0 baseline, which previously failed to converge (ep_err=0.47), now converges to ep_err=0.005 — equivalent to the pert_std=1 baseline. The center-out task provides sufficient gradient signal even without perturbation training.

### Results table (pert_scale=0, unperturbed eval)

| # | method | pert_std | loss_upd | ratio | vel(S=0.5) | ep_err | vel(S=0) | vel(S=1) | Δvel |
|---|--------|----------|----------|-------|------------|--------|----------|----------|------|
| 1 | std | 0 | no | — | 2.647 | 0.005 | 2.648 | 2.652 | +0.004 |
| 2 | std | 1 | no | — | 2.634 | 0.004 | 2.631 | 2.635 | +0.005 |
| 3 | std | 0 | yes | 0.3 | 1.654 | 0.037 | 1.657 | 1.645 | -0.012 |
| 4 | std | 1 | yes | 0.3 | 2.097 | 0.005 | 2.101 | 2.093 | -0.008 |
| 5 | std | 0 | yes | 0.5 | 2.196 | 0.004 | 2.189 | 2.199 | +0.010 |
| 6 | APT | 0 | yes | 0.3 | 1.657 | 0.038 | 1.656 | 1.650 | -0.006 |
| 7 | APT | 1 | no | — | 2.642 | 0.004 | 2.647 | 2.648 | +0.002 |
| 8 | APT | 1 | yes | 0.3 | 2.113 | 0.003 | 2.115 | 2.116 | +0.001 |
| 9 | std | 1 | yes | 0.5 | 2.219 | 0.003 | 2.215 | 2.223 | +0.008 |
| 10 | APT | 1 | yes | 0.5 | 2.223 | 0.004 | 2.219 | 2.218 | -0.001 |

### Robustness table (pert_scale=5, SISU=0.5)

| # | method | pert_std | loss_upd | ratio | lat_dev(p=5) | ep_err(p=5) |
|---|--------|----------|----------|-------|--------------|-------------|
| 1 | std | 0 | no | — | 0.015 | 0.013 |
| 2 | std | 1 | no | — | 0.014 | 0.010 |
| 7 | APT | 1 | no | — | 0.014 | 0.011 |
| 4 | std | 1 | yes | 0.3 | 0.016 | 0.011 |
| 8 | APT | 1 | yes | 0.3 | 0.013 | 0.010 |
| 9 | std | 1 | yes | 0.5 | 0.014 | 0.010 |
| 10 | APT | 1 | yes | 0.5 | 0.014 | 0.010 |

### Interpretation

- SISU velocity effect remains negligible (|Δvel| < 0.013) across all conditions — same null result as Phases 2-4.
- r=0.3 with pert_std=0 causes training degradation (models 3, 6: vel ~1.65, ep_err ~0.037) — the adaptive control cost is too aggressive without perturbation-derived gradient signal.
- r=0.5 performs better than r=0.3 overall: higher velocity and comparable robustness.
- APT provides marginal improvement over standard training in robustness metrics.
- Ratio sweep (r=0.1, 0.2, 0.4, 0.6) in progress to find optimal balance.

### Fine Ratio Sweep (r=0.10–0.20)

The fine sweep confirms a sharp transition between r=0.10 and r=0.12:

| ratio | vel(S=0.5) | ep_err | Δvel% |
|-------|-----------|--------|-------|
| 0.10 | 2.00 | 0.158 | +4.8% |
| 0.12 | 1.94 | 0.003 | +0.3% |
| 0.15 | 1.97 | 0.003 | +0.1% |
| 0.18 | 2.01 | 0.005 | -0.1% |
| 0.20 | 2.04 | 0.004 | +0.4% |

The SISU velocity effect exists ONLY when the model is at the edge of task failure (r=0.10, ep_err=0.158). At r≥0.12, reaching is competent and the effect vanishes. No intermediate "sweet spot" exists.

**Next approach: minimax training.** Rather than relying on random gusts (APT/CVaR), we are training a GaussianBumpAdversary that generates SISU-conditional force profiles via gradient ascent against the controller. The adversary is trained to maximise loss specifically at high SISU, creating a gradient pressure for the controller to increase velocity when SISU=1. Minimax training is currently in progress; results will be added in Phase 6.

## What This Means

1. **The running cost loss works.** It's the correct loss structure for reaching tasks in the graph architecture. Other modes need debugging.

2. **PAI-ASF models learn SISU-dependent accuracy modulation** — a real robustness signature, just not the one Crevecoeur & Scott highlighted.

3. **The velocity signature is harder to produce than expected.** Neither expected-cost optimization (standard backprop), tail-risk optimization (CVaR), nor adversarial optimization (APT with various hyperparameters) generates it. Possible remaining explanations:
   - The loss structure still doesn't sufficiently reward speed (no movement-time penalty)
   - The GRU architecture may need an explicit velocity-cost tradeoff mechanism
   - The point-mass dynamics may lack the biomechanical structure that produces co-contraction/impedance-based velocity changes in humans

4. **Adaptive control cost reduces peak speed and improves robustness but does not produce a meaningful SISU → velocity signature.** The ~3–4× weight increase is insufficient to create exploitable speed headroom. A much larger control penalty (or different loss structure such as an explicit movement-time penalty) may be needed.

5. **Translation-invariant center-out task fixes pert_std=0 convergence, but the SISU velocity null result persists across all conditions including center-out.** The task geometry does not explain the absence of the velocity signature.

## Files and Data

- `figures/` — interactive plotly HTML figures, referenced inline throughout this document.
- `models/` — saved trained models, training configs, and loss histories. One subdirectory per condition, each containing `config.json`, `trained_model.eqx`, and `train_history.eqx`.
