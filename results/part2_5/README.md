# Part 2.5 Experiment Results

## Overview

Comparison of loss functions and training objectives for PAI-ASF models.
Primary question: does any combination produce SISU-dependent peak velocity increase?

## Results Summary

### Phase 1: Loss Function Comparison (standard backprop)

| Condition | Loss mode | Status | Loss (initial→final) | Endpoint error | Peak speed | SISU→velocity |
|---|---|---|---|---|---|---|
| `running_cost_standard/` | Running cost from go cue | **Converged** | 32.3→2.9 | 0.007 | 3.33 | -0.3% |
| `softmin_standard/` | Goal-hit window (softmin) | **Diverged** | 5.0→843 | - | - | - |
| `default_standard/` | Structured mid/late ramps | **Diverged** | 14.9→885 | - | - | - |
| `combined_standard/` | Weak running + strong softmin | **Mediocre** | 10.7→5.0 | 0.15 | 2.06 | N/A |

**Winner:** `running_cost` — the only loss mode that trains stably with the graph architecture.

### Phase 2: Training Objective + Control Cost Variants

| Condition | Method | nn_output | Status | Loss | Ep error | Peak speed | SISU→velocity |
|---|---|---|---|---|---|---|---|
| `running_cost_cvar/` | CVaR 10% | 1e-5 | Converged (weak) | 32→17 | 0.109 | 3.18 | -0.9% |
| `running_cost_nn1e4/` | Standard | 1e-4 | Diverged | 32→714 | 0.78 | 2.63 | invalid |
| `running_cost_nn1e6/` | Standard | 1e-6 | Converged | 32→5.4 | 0.070 | 3.91 | -0.7% |
| `running_cost_apt/` | APT (adversarial) | 1e-5 | **Running** | - | - | - | - |

### Key Finding: Decoupled SISU Test

With fixed perturbation amplitude (scale=0.5), varying only the SISU input signal:

| SISU input | Peak velocity | Endpoint error | Interpretation |
|---|---|---|---|
| 0.00 | 3.330 | 0.0070 | Baseline |
| 0.50 | 3.327 | 0.0062 | **11% better accuracy** |
| 1.00 | 3.321 | 0.0069 | Slight accuracy improvement |

The network uses SISU for **accuracy modulation** (feedback gain changes) but NOT **velocity modulation** (trajectory shape changes). This confirms the LQG separation principle: expected-cost optimization changes gains, not trajectories.

## Figures

All in `figures/`:
- `fig_peak_velocity_by_sisu.html` — peak velocity vs SISU level, per condition
- `fig_endpoint_error_by_sisu.html` — endpoint error vs SISU level
- `fig_lateral_deviation_by_sisu.html` — max lateral deviation vs SISU
- `fig_loss_curves.html` — training loss over 10k iterations

## Data Files

Each condition directory contains:
- `config.json` — full training hyperparameters
- `trained_model.eqx` — trained model (equinox serialized, 5 replicates)
- `train_history.eqx` — loss history over training

## Feedbax Bugs Fixed

This experiment required 12 fixes to the feedbax develop branch (on `feature/fix-stale-experiments-imports`):
1. Stale `_experiments` imports (6 files)
2. `AbstractIntervenor` → `is_intervenor`
3. `Channel.change_input` reconstruction
4. Ensemble vmap broadcasting for StateIndex
5. Intervenor params single-pass processing
6. `_apply_inits` TimeSeriesParam skip
7. `filter_spec_leaves` Module→leaf expansion (**critical**: zero gradients)
8. `model.net` vs `model.nodes['net']` identity mismatch (**critical**: zero gradients)
9. `loss_reduction_fn` parameter for CVaR
10. Various training script fixes (where_train paths, nn_output CLI arg)

## What's Next

- **APT (adversarial perturbation training)**: the most promising remaining experiment. If minimax optimization produces velocity modulation where expected-cost doesn't, this confirms the theoretical prediction.
- **Movement-time penalty**: adding an accumulating cost that rewards early arrival
- **Adaptive control cost**: fixing the tracer leak in `loss_update_func` to enable dynamic control cost adjustment
