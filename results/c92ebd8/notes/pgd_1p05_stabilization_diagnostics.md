<!-- AUTO-GENERATED: pgd_1p05_stabilization_diagnostics -->
# PGD 1.05 Stabilization Task Diagnostics

This diagnostic reruns stabilization-task endpoint perturbation probes, not the reach-context perturbation-profile bank. AUC values are mean signed-direction-aligned absolute hand-position displacement over the post-onset window in `mm*s`.

| Row | Training | Level | Feedback AUC | Mechanical AUC | Command AUC | Process-force AUC |
|---|---|---:|---:|---:|---:|---:|
| `open_loop_small` | `no_pgd_open_loop` | small | 7.195 | 0.8195 | 1.137 | 0.5017 |
| `open_loop_moderate` | `no_pgd_open_loop` | moderate | 7.25 | 0.74 | 1.012 | 0.4681 |
| `open_loop_stress` | `no_pgd_open_loop` | stress | 8.142 | 0.616 | 0.7458 | 0.4861 |
| `small` | `pgd_1p05` | small | 7.28 | 0.8208 | 1.118 | 0.5233 |
| `moderate` | `pgd_1p05` | moderate | 7.524 | 0.7387 | 0.9321 | 0.5453 |
| `stress` | `pgd_1p05` | stress | 7.158 | 0.6559 | 0.7698 | 0.5419 |

## Interpretation

Mean PGD/no-PGD feedback AUC ratio: `0.976` (approximately_unchanged).
Mean PGD/no-PGD mechanical AUC ratio: `1.02` (approximately_unchanged).
Qualitative locus: `neither`.

Definitions: feedback AUC averages the position, velocity, and force/filter false-feedback offset families. Mechanical AUC averages non-feedback `command_input_pulse` and `process_epsilon_force_state_xy` pulses.

Caveat: as in issue 87424a4, rows are labeled as wash-in endpoint responses unless the strict drift threshold is met; the API uses deterministic prefix-equivalent fan-out rather than a literal hidden-state snapshot fork.
<!-- /AUTO-GENERATED -->
