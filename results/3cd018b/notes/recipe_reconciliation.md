# 3cd018b Recipe Reconciliation Audit

Date: 2026-07-08

This note audits the eight 3cd018b row recipes against the synced RunPod
`float32_rebound_specs` copies that the 2026-07-07 launch consumed.

## Summary

- Pod launch source: `_artifacts/3cd018b/runpod_remote_run_dir/main_20260707T164908Z/launch_row.sh` points each row at `/workspace/feedbax_runs/3cd018b-adaptive-target-matrix/float32_rebound_specs/results/3cd018b/runs/${row}.json`.
- The original launch-lock branch `feature/3cd018b-adaptive-target-matrix` has zero `method_payload` diffs against the pod-rebound spec for all eight rows.
- Every launch-lock row differs from the pod-rebound graph in exactly two leaves: `inline/nodes/mechanics/params/B/4/0` and `inline/nodes/mechanics/params/B/5/1`, from `0.1515151560306549` to `0.1515151411294937`.
- This issue branch started from the 81e3d8d path-normalization branch, where only the adopted `ramp3500_to1000` row was present. That adopted ramp recipe had the method drift listed below; the other seven row recipes were absent from this branch before reconciliation.
- Reconciled tracked recipes are copied from the pod-rebound specs with `spec_dir` / `tracked_spec_dir` normalized to repo-relative `results/3cd018b/runs/<row>` values.
- Phase-program parity: current `adaptive_epsilon_method_contract().phase_program` hash is `a3060c554b3cc7bd44e81217b7a96fa26d6967af5fc71a1d95436273e324565c`, matching the stored checkpoint binding.

## Per-Row Audit

| Row | Launch-lock method payload vs pod rebound | Graph constant diff | Pre-reconciliation issue-branch state |
|---|---:|---|---|
| `hold3500_to1000` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `hold1750_to1000` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `hold3500_to250` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `hold1750_to250` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `const1750` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `const1000` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `const250` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | absent from issue branch |
| `ramp3500_to1000` | 0 | `inline/nodes/mechanics/params/B/4/0` `0.1515151560306549` -> `0.1515151411294937`; `inline/nodes/mechanics/params/B/5/1` `0.1515151560306549` -> `0.1515151411294937` | present but method-drifted after adoption |

## Current-Branch Ramp Method Drift

Before this reconciliation, `results/3cd018b/runs/ramp3500_to1000.json` on the issue branch differed from the pod-rebound `method_payload` at these leaves:

- `payload/checkpointing/resume`: `true` -> `false`
- `payload/config/adaptive_epsilon_controller_training_mode`: `loss_blend` -> `epsilon_scaled_outer_training`
- `payload/config/adaptive_epsilon_damage_anneal_batches`: `5000` -> `2500`
- `payload/config/adaptive_epsilon_damage_ramp_batches`: `2500` -> `1000`
- `payload/config/adaptive_epsilon_outer_weight_ramp_batches`: `2500` -> `1000`
- `payload/config/allow_fresh_start`: `false` -> `<missing>`
- `payload/config/policy_adversary_radius_15cm`: `0.004545500088363065` -> `null`
- `payload/config/policy_adversary_radius_source`: `effective_020a65b_pgd_training_radius` -> `null`
- `payload/config/spec_dir`: `results/3cd018b/runs/ramp3500_to1000` -> `<absolute feature worktree>/results/3cd018b/runs/ramp3500_to1000`
- `payload/controller_training_mode`: `loss_blend` -> `epsilon_scaled_outer_training`
- `payload/damage_schedule/anneal_batches`: `5000` -> `2500`
- `payload/damage_schedule/ramp_batches`: `2500` -> `1000`
- `payload/outer_adversarial_weight/applies_to`: `optimized_direct_epsilon_loss_only` -> `optimized_direct_epsilon_channel_scale_for_controller_rollout`
- `payload/outer_adversarial_weight/ramp_batches`: `2500` -> `1000`

After reconciliation, the path leaf is intentionally stored as the repo-relative `results/3cd018b/runs/ramp3500_to1000` rather than the pod/worktree absolute path.

## Likely Drift Mechanism

The pod run consumed the rebound specs staged under `/workspace/feedbax_runs/.../float32_rebound_specs`, not the later mainline/adopted ramp recipe. The original eight-row launch-lock specs already had the same scientific `method_payload` as the pod copies; the rebound step only corrected the runtime graph constants under float32. The drift observed on this issue branch came from the mainline resume/adoption path for `ramp3500_to1000` after the launch-lock branch: the branch only carried the adopted ramp recipe, and that recipe was regenerated or refreshed with `loss_blend`, longer ramp/anneal counts, `resume=true`, and policy-adversary radius fields that were not in the launched pod payload. The seven non-ramp row recipes simply had not been carried onto this branch before reconciliation.

## Lineage Policy

Do not try to make old checkpoint binding hashes pass by restoring absolute worktree paths or old payload projections. Future resume/fork work for these rows should use `fork_checkpoint_transaction` with `allow_new_lineage_override`, record why a new lineage is being created, and treat the reconciled portable recipe as the source of truth for the new fork lineage.
