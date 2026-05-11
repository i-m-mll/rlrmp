# Cross-method induced-gain comparison: Part 2.5 baselines vs flavor-(b)

**Issue.** `74bfd86`
**Branch.** `feature/induced-gain-flavor-b`
**Companion run-spec.** [`run.json`](../runs/induced_gain_flavor_b/run.json)
**Date.** 2026-05-08

## Setup

- Plant: rlrmp regime, `linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)`.
- Cost: `cost_schedule_from_spec(CostSpec(n_steps=200))` (`Q_f=1.0`).
- Reach: `(0,0) -> (0.15, 0.0)` (15 cm forward), held mid-movement (hold=0, go=1).
- SISU: 0.5.
- Algorithm: power iteration only (3 restarts, max_iter=600, **rtol=1e-06**, post-probe canonical).
- Riccati baseline: `gamma_star = 0.013749`.
- Replicate hygiene rule: flag `gamma > 10x` from group median (or `< median / 10`); flagged replicates excluded from per-eta medians.

Pre-registered headline metric: **`gamma_sd x qr_cost`** (induced gain on the
`structural_da` w channel against cost-matched z). This is the channel that
was uniformly large (~150–170) across all flavor-A trained controllers in
the first run — flavor-B training is the manipulation that should shift it
if the (a) ⊊ (b) thesis is empirically supported.

## Table 1 — Headline `gamma_sd x qr_cost`

Flavor-A first-run values from `results/c723082/notes/induced_gain_first_run.md`
(single replicate per group, replicate 0). Flavor-B values from this run; per-eta
median + MAD across **non-outlier** replicates (3 seeds × 5 reps − flagged).

| Method / config | `gamma_sd` (median) | spread | `gamma_sd / gstar` | n_kept / n_outliers | Notes |
|---|---|---|---|---|---|
| `baseline_standard_12k` | 148.50 | (single rep) | 10801 | 1/0 | flavor-A baseline (warmup only) |
| `vanilla_single` | 169.26 | (single rep) | 12311 | 1/0 | flavor-A baseline (no adversary) |
| `vanilla_pop5` | 164.64 | (single rep) | 11975 | 1/0 | flavor-A baseline (no adversary) |
| `minimax_single_seed0` | 153.35 | (single rep) | 11153 | 1/0 | flavor-A APT minimax |
| `minimax_single_seed1` | 162.82 | (single rep) | 11842 | 1/0 | flavor-A APT minimax |
| `minimax_single_seed2` | 162.92 | (single rep) | 11849 | 1/0 | flavor-A APT minimax |
| `mult_single (rep0; degen.)` | 5863.87 | (single rep) | 426491 | 1/0 | documented degenerate rep-0; first-run replaced with rep-2 |
| `mult_single (rep2; replacement)` | — | — | — | 1/0 | flavor-A multiplicative (corrected) |
| `mult_pop5` | 165.06 | (single rep) | 12005 | 1/0 | flavor-A multiplicative |
| `ratio03_single` | 163.39 | (single rep) | 11884 | 1/0 | flavor-A ratio03 |
| `ratio03_pop5` | 164.53 | (single rep) | 11967 | 1/0 | flavor-A ratio03 |
| **`flavor_b_eta0.03`** (3 seeds × 5 reps) | **152.53** | MAD=9.76 (range 137.50–1001.72) | 11094 | 15/0 | flavor-B (this run) |
| **`flavor_b_eta0.10`** (3 seeds × 5 reps) | **156.22** | MAD=9.95 (range 124.45–266.02) | 11362 | 15/0 | flavor-B (this run) |
| **`flavor_b_eta0.30`** (3 seeds × 5 reps) | **154.78** | MAD=14.54 (range 119.23–306.56) | 11257 | 15/0 | flavor-B (this run) |

Flavor-A cross-method median (excluding `mult_single` rep-0 degenerate): **163.39**.

**Headline ratio**: flavor-B `gamma_sd` median = **154.78**, 
flavor-A `gamma_sd` median = **163.39**, ratio = **0.947**.

## Table 2 — Auxiliary channels (`gamma_af`, `gamma_sp`)

| Method / config | `gamma_af` | `gamma_sp` |
|---|---|---|
| `baseline_standard_12k` | 0.1237 | 2.61 |
| `vanilla_single` | 0.2481 | 4.33 |
| `vanilla_pop5` | 0.1458 | 4.41 |
| `minimax_single_seed0` | 0.1446 | 2.14 |
| `minimax_single_seed1` | 0.1563 | 1.32 |
| `minimax_single_seed2` | 0.1556 | 1.45 |
| `mult_single (rep0; degen.)` | 15.6446 | 605.39 |
| `mult_single (rep2; replacement)` | 0.1834 | — |
| `mult_pop5` | 0.1789 | 4.22 |
| `ratio03_single` | 0.1709 | 2.64 |
| `ratio03_pop5` | 0.1585 | 2.10 |
| **`flavor_b_eta0.03`** | 0.1658 (MAD=0.0440, n=13/2) | 1.39 (MAD=0.64, n=12/3) |
| **`flavor_b_eta0.10`** | 0.1320 (MAD=0.0205, n=15/0) | 0.95 (MAD=0.29, n=14/1) |
| **`flavor_b_eta0.30`** | 0.1403 (MAD=0.0244, n=15/0) | 1.25 (MAD=0.41, n=12/3) |

## Table 3 — Per-(eta, seed) flavor-B detail

| Group | n_kept / n_outliers | `gamma_sd` median | `gamma_af` median | `gamma_sp` median |
|---|---|---|---|---|
| `flavor_b_eta0.03__seed_0` | 5/0 | 155.91 | 0.2098 | 2.78 |
| `flavor_b_eta0.03__seed_1` | 5/0 | 150.02 | 0.1310 | 0.88 |
| `flavor_b_eta0.03__seed_2` | 5/0 | 151.75 | 0.2023 | 3.49 |
| `flavor_b_eta0.10__seed_0` | 5/0 | 151.83 | 0.1266 | 0.88 |
| `flavor_b_eta0.10__seed_1` | 5/0 | 156.22 | 0.1320 | 1.44 |
| `flavor_b_eta0.10__seed_2` | 5/0 | 159.72 | 0.1512 | 0.89 |
| `flavor_b_eta0.30__seed_0` | 5/0 | 154.78 | 0.1403 | 0.86 |
| `flavor_b_eta0.30__seed_1` | 5/0 | 165.65 | 0.1336 | 1.38 |
| `flavor_b_eta0.30__seed_2` | 5/0 | 153.69 | 0.2148 | 2.61 |

## eta_max trend (flavor-B sweep)

Per-eta median across all kept replicates (after hygiene exclusion):

| eta_max | `gamma_sd` median | `gamma_af` median | `gamma_sp` median |
|---|---|---|---|
| 0.03 | 152.53 | 0.1658 | 1.39 |
| 0.10 | 156.22 | 0.1320 | 0.95 |
| 0.30 | 154.78 | 0.1403 | 1.25 |
