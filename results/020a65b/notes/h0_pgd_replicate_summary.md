# H0 PGD Replication Summary

Issue: `020a65b`

Scope: two local CPU h0 rows at `lr=3e-3`, both with target-relative multi-target
training, force/filter feedback, calibrated small perturbation training, full
Q/R/Q_f loss, batch 64, 5 replicates, and a 12000-batch contract. Each row first
passed a 1000-batch sanity gate before resuming to 12000.

Runs:

| row | run id | completed batches |
|---|---|---:|
| h0 no-PGD | `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 12000 |
| h0 PGD-OFB | `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 12000 |

## Main Comparison

| metric | h0 no-PGD | h0 PGD-OFB | interpretation |
|---|---:|---:|---|
| shared-rollout GRU/extLQG full-QRF ratio | 17.6337 | 4.3461 | PGD greatly improves this h0 comparison lens. |
| standard clean action mismatch ratio | 1.6977 | 1.9843 | PGD does not improve the clean action-mismatch sidecar. |
| same-channel optimized epsilon delta cost | 847.979 | 299.986 | PGD lowers the frozen-controller worst-case epsilon audit delta. |
| H-infinity phenotype evidence | available | available | Explicit no-PGD -> PGD pairing is materialized. |

The robustness signature does appear in the same sense as the 020a65b PGD lane:
the PGD h0 row is much closer to extLQG on the shared-rollout objective
comparator and is less vulnerable in the same-channel worst-case epsilon audit.
It is not a full formal H-infinity claim, and it is not a clean improvement on
every sidecar.

## Caveats

- Standard certificates remain `partial_standard_certificate_blocked` for both
  h0 rows. The response-map comparison is explicitly blocked because these rows
  use a 6D delayed position/velocity plus force/filter feedback contract, while
  the current analytical output-feedback reference is 8D and has no approved
  6D-to-8D projection.
- Broad-epsilon attribution is `not_applicable` for both rows; these runs use
  the PGD inner maximizer path, not the sampled broad-epsilon training adapter
  that attribution removes/replays.
- Training diagnostics were restored from saved chunk histories after a
  resume-sidecar final-write bug dropped loss arrays. The code now preserves
  already-complete arrays on no-op resume writes and normalizes replicate-major
  resumed chunks before stitching.

## Materialized Outputs

- `results/020a65b/notes/gru_postrun_materialization_h0_pgd_bank_two_rows_validation_selected.json`
- `results/020a65b/notes/gru_standard_certificates_h0_pgd_bank_two_rows_validation_selected.md`
- `results/020a65b/notes/objective_comparator_h0_pgd_bank_two_rows_validation_selected.md`
- `results/020a65b/notes/gru_map_error_decomposition_h0_pgd_bank_two_rows_validation_selected.md`
- `results/020a65b/notes/gru_feedback_ablation_h0_pgd_bank_two_rows_validation_selected.md`
- `results/020a65b/notes/gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected.md`
- `results/020a65b/notes/gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected_calibrated.md`
- `results/020a65b/notes/gru_perturbation_response_norm_plots_h0_pgd_bank_two_rows_validation_selected_calibrated.md`
- `results/020a65b/notes/gru_worst_case_epsilon_audit_h0_pgd_bank_two_rows_validation_selected.md`
- `results/020a65b/notes/h0_pgd_bank_two_rows_validation_selected_broad_epsilon_attribution.md`
- `results/020a65b/notes/hinf_phenotype_sidecar_h0_pgd_bank_two_rows_validation_selected.md`
