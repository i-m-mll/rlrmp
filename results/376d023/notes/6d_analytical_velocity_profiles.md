<!-- AUTO-GENERATED: six_d_velocity_profiles -->
# 6D analytical output-feedback comparators

Generated the no-integrator 6D extLQG and output-feedback H-infinity analytical models under the same delayed force-filter feedback contract used by the h0 rows.

| Row | Peak mean forward velocity (m/s) | Time of peak (s) | Samples |
|---|---:|---:|---:|
| 6D extLQG analytical | 0.731475 | 0.16 | 320 |
| 6D output-feedback H-infinity analytical | 0.760662 | 0.16 | 320 |
| 020a65b h0 no-PGD | 0.731686 | 0.16 | 320 |
| 020a65b h0 PGD | 0.778829 | 0.16 | 320 |

Interpretation: The 6D H-infinity output-feedback arm preserves the expected robustification signature relative to 6D extLQG: higher/faster nominal forward velocity under the same force-filter feedback contract.

Artifacts:
- Plot: `_artifacts/376d023/figures/6d_analytical_velocity_profiles/velocity_profile_overlay.html`
- Summary: `_artifacts/376d023/figures/6d_analytical_velocity_profiles/figure_summary.json`
- Figure spec: `results/376d023/figures/6d_analytical_velocity_profiles/spec.json`
- Distillation plan: `results/376d023/runs/proposed_h0_no_pgd_distillation_6d_teacher.json`

Rows used:
- h0 no-PGD: `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`
- h0 PGD: `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64`

<!-- /AUTO-GENERATED -->
