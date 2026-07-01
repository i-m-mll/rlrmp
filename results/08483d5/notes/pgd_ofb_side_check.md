<!-- AUTO-GENERATED: pgd_ofb_side_check -->
# PGD OFB Side Check

## Scope

This local side check answers the older-PGD questions for issue `08483d5`. It does not launch training, update controller weights, request auth, push, or treat historical cap/radius/trust-region values as new defaults.

Checkpoint policy: validation-selected per replicate using sparse history. Batch convention: 64 repeats of the fixed +x nominal 15 cm validation reach. Noise convention: paired clean/adversarial damage uses identical stochastic rollout keys; nominal peak velocity pools repeated stochastic validation trials across replicates.

## Nominal Peak Velocity

| row | dim | peak m/s | delta vs c92 no-PGD | percent | run spec |
|---|---:|---:|---:|---:|---|
| `open_loop_moderate` | 6D | 0.731086 | +0.000000 | +0.00% | `results/c92ebd8/runs/open_loop_moderate.json` |
| `moderate_pgd_ofb1p05` | 6D | 0.757901 | +0.026815 | +3.67% | `results/c92ebd8/runs/moderate_pgd_ofb1p05.json` |
| `moderate_pgd_ofb1p4` | 6D | 0.765344 | +0.034258 | +4.69% | `results/c92ebd8/runs/moderate_pgd_ofb1p4.json` |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | 8D | 0.729827 | -0.001259 | -0.17% | `results/33b0dcb/runs/h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64.json` |

The 6D c92 no-PGD baseline is the direct same-family baseline for the two older PGD rows. The 33b0dcb const_band16 row is shown as live baseline context, but it is 8D and therefore not the dimensional match for the c92 PGD rows.

## Recomputed PGD Damage

| row | radius | gamma factor | clean cost | adversarial cost | paired damage | damage/reference | boundary fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `moderate_pgd_ofb1p05` | 0.001751332567 | 1.05 | 4515.5130 | 5254.0763 | 738.5633 | 0.1205 | 1.000 |
| `moderate_pgd_ofb1p4` | 0.004545011587 | 1.4 | 4683.2999 | 8180.7530 | 3497.4532 | 0.5704 | 1.000 |

Cost definition: full no-integrator C&S Q/R/Q_f task cost for the 6D PGD rows; the disturbance penalty is not subtracted. The reference damage is the paired nominal-noise output-feedback damage from the first 08483d5 check (6131.6907).

## Provenance

- `moderate_pgd_ofb1p05` uses the historical output-feedback rollout radius `ofb_6d_no_integrator_gamma_1p05_rollout_radius`; this is provenance only.
- `moderate_pgd_ofb1p4` uses the historical output-feedback rollout radius `ofb_6d_no_integrator_gamma_1p4_rollout_radius`; this is provenance only.
- Existing prior `ofb1p4` damage path: `results/08483d5/notes/gru_pgd_damage_sanity.json`.
- Existing c92 validation-selected diagnostic table: `results/c92ebd8/notes/output_feedback_budget_diagnostics.md`.

## Uncertainty

- The c92 rows are 6D no-integrator GRUs; the 33b0dcb const_band16 baseline is 8D and is reported only as baseline-context provenance.
- The adversaries here were recomputed by a local 10-step projected-gradient ascent on the frozen nominal batch; they are not stored training adversaries.
- Both PGD damage estimates are small frozen-batch diagnostics, not new training runs or launch recommendations.
- The historical OFB radii are reported only as provenance for these older rows, not as recommended defaults.

## Outputs

- JSON sidecar: `results/08483d5/notes/pgd_ofb_side_check.json`
- Script: `results/08483d5/scripts/compute_pgd_ofb_side_check.py`
<!-- /AUTO-GENERATED -->
