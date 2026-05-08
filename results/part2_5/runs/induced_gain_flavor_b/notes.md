# Induced-gain analyser on flavor-(b) (LinearDynamicsAdversary) checkpoints

**Issue.** `74bfd86`
**Branch.** `feature/induced-gain-flavor-b`
**Date.** 2026-05-08

## Goal

Run `||T_{w → z}||_∞` on the 9 flavor-(b) trained controllers from `c723082`
(eta_max ∈ {0.03, 0.10, 0.30} × seed ∈ {0, 1, 2}, 5 vmap replicates each =
45 controllers), and compare cross-method to the Part 2.5 first-run (flavor-A)
baselines on the pre-registered headline metric `γ_sd × qr_cost`.

## Setup

Same plant / cost / reach / SISU / algorithm as `induced_gain_first_run`,
**except**: rtol=1e-6 (canonical post-probe), all 5 internal replicates
analysed per config. Hygiene rule: flag γ replicates with `ratio > 10×` from
group median (or `< median / 10`); flagged replicates excluded from per-eta
medians.

- Plant: rlrmp regime, `linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)`.
- Cost: `cost_schedule_from_spec(CostSpec(n_steps=200))`.
- Reach: `(0,0) → (0.15, 0.0)`.
- SISU: 0.5.
- Power iteration: 3 restarts, max_iter=600, **rtol=1e-6**.
- Riccati baseline: γ⋆ = 0.013749.

## Headline result

Pre-registered metric **`γ_sd × qr_cost`**:

| Method bucket | median γ_sd | n |
|---|---|---|
| flavor-A baselines (Part 2.5 first run) | **163.39** | 9 (excl. mult_single rep0 degen.) |
| flavor-B (this run, 3 etas pooled) | **154.78** | 45 |

**Ratio = 0.947** (≈ 5% reduction).

eta_max trend in γ_sd (per-eta median): 152.53 (eta=0.03) → 156.22 (eta=0.10) → 154.78 (eta=0.30) — **flat**, NOT a U-shape. This does NOT line up with the U-shape in training ctrl_loss reported on `c723082` (4.75 → 5.12 → 4.70).

## Verdict on the (a) ⊊ (b) thesis (this metric, this geometry)

**Not supported** by `γ_sd × qr_cost` on this single canonical reach. The induced-gain on the structural-`ΔA` channel is essentially indistinguishable between flavor-A and flavor-B trained controllers, and shows no eta_max dependence.

This is a partial signal, not a full refutation: the metric measures closed-loop sensitivity to unstructured ΔA at one mid-movement linearisation, not the full repertoire of state-coupled flavor-B perturbations the controller has seen during training. Possible reasons γ_sd doesn't shift:

1. **Operator-norm metric is direction-insensitive**: γ_sd reports the worst-case direction; flavor-B training may shift attenuation in *specific* (training-relevant) ΔA directions while leaving the worst-case direction unchanged.
2. **Single linearisation point**: γ_sd at hold=0 / go=1 / sisu=0.5 may not be where flavor-B training had effect.
3. **Low effective adversary intensity**: even at η=0.30 (the largest budget tested), the controller may not need to allocate much robustness; the additive-force first-run γ_af 0.12-0.25 across all flavor-A methods suggests the closed loop is already operating well below H∞ optimum.

## Auxiliary channels (`γ_af`, `γ_sp`)

- **γ_af (additive force)**: flavor-B medians 0.13-0.17, similar to flavor-A baselines (0.12-0.25). No clear advantage.
- **γ_sp (sensory perturbation)**: flavor-B medians 0.95-1.39, **lower** than flavor-A vanilla (4.33-4.41) and comparable to or lower than flavor-A minimax (1.32-2.14). Flavor-B training appears to confer sensory-perturbation robustness comparable to or better than the best flavor-A method.

## Outliers and methodology notes

- **Hygiene rule (10× threshold) flagged 0 γ_sd replicates** out of 45 — the 5-rep median is robust enough that no individual rep was >10× from its group median for the headline channel. Auxiliary channels showed 5/45 γ_af outliers and 8/45 γ_sp outliers, which were excluded from per-eta medians.
- Some replicates show **moderately elevated γ_sd** (370, 443, 1001, 306, 247, 266) — between 1.5× and 6.6× group median. These are kept under the 10× rule and therefore widen the within-group MAD without changing the median materially. The 1001.7 value (eta=0.03 seed_2 rep 4) is the most extreme; coincides with γ_sp = 39.1 and γ_af = 2.07 on the same replicate (consistent with a single-replicate degenerate fixed point analogous to the documented `mult_single` rep-0 in the first run).
- All groups loaded successfully; no replicate failed adapter construction.
- All 9 × 5 × 3 = 135 channel-replicate combinations converged in power iteration.

## Next steps

- **Sweep over reach geometries / SISU values**: γ_sd at one canonical reach is too narrow.
- **Worst-case-direction projection**: compute the ΔA direction that achieves γ_sd at each replicate and check whether flavor-B-trained controllers have moved attenuation away from the directions flavor-B training visited.
- **The eta_max trend (flat in induced gain, U in training loss)** suggests the U-shape on `c723082` is a saddle-dynamics / reporting artifact, not a robustness inversion. The induced-gain analyser does not see the eta=0.10 anomaly.

## Cross-references

- Spec: [`run.json`](run.json)
- Heavy outputs: `_artifacts/part2_5/runs/induced_gain_flavor_b/<group>/gains.json` + `summary.json`
- Cross-method comparison: [`results/part2_5/induced_gain_flavor_b/cross_method_comparison.md`](../../induced_gain_flavor_b/cross_method_comparison.md)
- Source training: issue `c723082`
- Source analyser: issue `74bfd86` (this issue)
- First-run flavor-A baselines: `results/part2_5/runs/induced_gain_first_run/notes.md`
