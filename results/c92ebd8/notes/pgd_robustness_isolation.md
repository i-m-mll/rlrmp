<!-- AUTO-GENERATED: pgd_robustness_isolation -->
# PGD robustness isolation

This note uses `stabilization task` for the current endpoint perturbation diagnostic. It does not launch training or change the existing c92 stabilization figure layout.

## Stabilization task

| Source | Row | Training | Level | Feedback AUC | Mechanical AUC | Command AUC | Process-force AUC |
|---|---|---|---:|---:|---:|---:|---:|
| 020a65b | `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 020a65b H0 no-PGD | small | 7.141 | 7.316 | 0.9238 | 13.71 |
| 020a65b | `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 020a65b H0 PGD-OFB gamma_factor=1.4 | small | 7.478 | 0.8014 | 0.9563 | 0.6464 |
| c92ebd8 | `open_loop_small` | no_pgd_open_loop | small | 7.195 | 0.8195 | 1.137 | 0.5017 |
| c92ebd8 | `open_loop_moderate` | no_pgd_open_loop | moderate | 7.25 | 0.74 | 1.012 | 0.4681 |
| c92ebd8 | `open_loop_stress` | no_pgd_open_loop | stress | 8.142 | 0.616 | 0.7458 | 0.4861 |
| c92ebd8 | `small` | pgd_1p05 | small | 7.28 | 0.8208 | 1.118 | 0.5233 |
| c92ebd8 | `moderate` | pgd_1p05 | moderate | 7.524 | 0.7387 | 0.9321 | 0.5453 |
| c92ebd8 | `stress` | pgd_1p05 | stress | 7.158 | 0.6559 | 0.7698 | 0.5419 |

The current stabilization-task diagnostic directionally reproduces the older feedback-up/mechanical-down PGD pattern on the 020a65b small H0 pair: feedback ratio `1.047` and mechanical ratio `0.110`. The feedback increase is small and below a strict 5% material-effect threshold, while the mechanical reduction is large. The c92 PGD 1.05 mean ratios stay approximately unchanged: feedback `0.976` and mechanical `1.022`.

## Matched Reach Context

| Source | Level | Training | Matched families | Sensory AUC | Non-sensory AUC | Peak dx/OL |
|---|---:|---|---:|---:|---:|---:|
| 020a65b | small | no-PGD | 8 | 0.301 | 3.94 | missing |
| 020a65b | small | PGD gamma_factor=1.4 | 8 | 0.263 | 2.43 | missing |
| c92ebd8 | moderate | no-PGD | 8 | 0.683 | 1.97 | 0.468 |
| c92ebd8 | moderate | PGD gamma/gamma_star=1.05 | 8 | 0.608 | 1.84 | 0.473 |
| c92ebd8 | small | no-PGD | 8 | 0.675 | 2.2 | 0.494 |
| c92ebd8 | small | PGD gamma/gamma_star=1.05 | 8 | 0.598 | 2 | 0.489 |
| c92ebd8 | stress | no-PGD | 8 | 0.561 | 1.62 | 0.431 |
| c92ebd8 | stress | PGD gamma/gamma_star=1.05 | 8 | 0.504 | 1.42 | 0.435 |

Matched-family aggregation preserves the qualitative split: 020a65b shows a much larger PGD/non-PGD reach-context AUC reduction than c92 PGD 1.05, even after excluding the 020a65b-only integrator-epsilon family.

Matched-family exclusions: `process_epsilon/process_epsilon_integrator_xy` is present in 020a65b but excluded because the current c92 contract is 6D/no-integrator.

## Remaining Plausible Factors

- 020a65b uses the older 8D C&S coordinate basis with process integrator rows; c92 uses a 6D no-integrator process contract.
- 020a65b PGD sidecars record broad full-state epsilon PGD with gamma_factor=1.4; c92 PGD uses gamma/gamma_star=1.05.
- The perturbation training/evaluation family and timing contracts differ: 020a65b is a wider reach-context bank, while the c92 stabilization task isolates endpoint feedback/mechanical probes.
- Feedback-scale and calibration provenance differ between the older open-loop/proprio-calibrated rows and the current c92 calibrated open-loop matrix.
- The current stabilization task has only a small-level 020a65b pair; moderate/stress levels cannot be fabricated for that historical run.

## Outputs

- `summary_json`: `results/c92ebd8/notes/pgd_robustness_isolation.json`
- `summary_markdown`: `results/c92ebd8/notes/pgd_robustness_isolation.md`
- `summary_csv`: `results/c92ebd8/notes/pgd_robustness_isolation_summary.csv`
- `matched_reach_csv`: `results/c92ebd8/notes/pgd_robustness_isolation_matched_reach_families.csv`
- `bulk_dir`: `_artifacts/c92ebd8/stabilization_diagnostics/pgd_robustness_isolation`
- `figure_spec`: `results/c92ebd8/figures/pgd_robustness_isolation_stabilization_responses/spec.json`
- `figure_artifact_dir`: `_artifacts/c92ebd8/stabilization_diagnostics/pgd_robustness_isolation/stabilization_responses`
<!-- /AUTO-GENERATED -->
