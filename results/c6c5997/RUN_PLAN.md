# Stage-2 Adaptive-Epsilon LR-Continuity Matrix

Issue: c6c5997

This matrix asks whether continuation-local learning-rate shape changes the
adaptive-epsilon response after the harmonized nominal baseline. The three
rows fork independently from the immutable repaired batch-12,000 custody
source. Each row is one vectorized ensemble with five independently initialized
replicas; the replicas are not separate matrix rows.

## Shared source and task

- Source recipe: `results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json`.
- Source custody root: `_artifacts/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64/retry_sources/custody_repaired_12000_r3`.
- Source transaction: `tx-b15820671bf943768cc082183d5b295b`.
- Source progress: 12,000 completed training batches. Program step 24 and
  barrier ordinal 24 remain checkpoint coordinates, not batch totals.
- Target progress: 16,500 completed training batches, a 4,500-batch
  continuation.
- Canonical task identity: `sha256:66d64094620cb89393c1e249c52bcaeb17f744497fa0ea6d3eaeb5717c079456`.

The matrix base carries the actual baseline `game_card` and
`perturbation_training` payloads. Matrix and row metadata carry labels derived
from those payloads; the fork gate recomputes and compares them before writing
any target checkpoint.

## Locked rows

| Row | Continuation-local LR schedule | LR mode | Target total |
|---|---|---|---:|
| `flat_3e-5` | `3e-5` throughout using warmup-cosine with unit warmup fraction and cosine alpha | `restart` | 16,500 |
| `rewarm_3e-4` | `3e-5` to `3e-4`, cosine back to `3e-5`, then hold | `restart` | 16,500 |
| `rewarm_3e-3` | `3e-5` to `3e-3`, cosine back to `3e-5`, then hold | `restart` | 16,500 |

All schedules use 1,000 warmup batches and 3,500 scheduled continuation steps;
the final 1,000 continuation batches hold the terminal `3e-5` rate. The
matrix declaration and every adaptive method payload both declare `restart`;
a missing or unequal payload declaration fails the prelaunch gate.

## Shared adaptive contract

- Direct-epsilon soft-energy inner maximization, 12 Adam steps at `2e-5`.
- Fixed-ratio damage setpoint with numerator convention `excess`.
- Epsilon application ramps from zero to one over the first 1,000 continuation
  batches. The perturbation-training bank is orthogonal to that scale.
- Clamp interval 50 batches, damage EMA alpha 0.1, eta 0.2,
  `max_log_step=0.15`, and `deadband_frac=0.03`.
- Application-ramp freeze is off. Clamp and damage EMA are active from the
  first continuation batch; the ramp is diagnostics-only.
- Analytical lambda seed: `281032999.21861446`; lambda minimum:
  `281032.9992186145`, exactly `0.001` times that seed.
- Radius, energy, and safety caps remain disabled.

## Pending seam-derived substitution

The tracked recipes deliberately contain the fail-closed sentinel
`__R_STAR_PENDING_SEAM_BASELINE_FINAL_QUARTER__`. After the isolated seam probe
passes, replace every sentinel with values derived from:

```text
raw_ratio = 1024 / mean(clean loss over baseline batches 9001-12000)
R* = raw_ratio rounded to two significant figures
```

The matrix `ratio_setpoint` metadata must retain the raw numerator `1024`,
numerator convention `excess`, denominator-window label
`baseline_final_quarter`, the numeric denominator mean, and rounded `R*`.
The three row damage schedules and their config mirrors must all receive the
same numeric rounded `R*`. The fork gate is expected to reject the sentinel
before any checkpoint write.

## Prelaunch gates

1. Validate the three materialized Feedbax `TrainingRunSpec` rows.
2. Recompute and compare canonical task identity for source and every row.
3. Require the tracked clamp contract and exact analytical-seed-relative
   lambda minimum.
4. Require declared/payload LR-continuation equality and print multiple LR
   schedule points through the executor builder.
5. Fork the real repaired source to isolated row targets and verify source to
   target barrier mapping, target manifest and slot totals of 16,500, all six
   optimizer-history horizons of 16,500, and adaptive target-only provenance.
6. Print the full R-star derivation components before training.

Only the supplied secure RTX 5090 run authority permits the billable launch.
No GPU class or cloud-tier substitution is part of this lock.
