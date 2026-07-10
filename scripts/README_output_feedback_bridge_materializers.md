# Output-Feedback Bridge Materializers

This note marks which output-feedback bridge scripts are active Feedbax-backed
spec runners and which ones are retained as historical or provenance-only
writers. New bridge custody work should add `AnalysisRunSpec` recipes and
`rlrmp/output_feedback_bridge` bundle entries instead of extending ad hoc
note/manifest writers.

## Active Spec Runners

| script | diagnostic | custody |
|---|---|---|
| `materialize_output_feedback_rollout_recovery.py` | `7a459bb` rollout-recovery bridge diagnostic | Runs `rlrmp.output_feedback_bridge.rollout_recovery` through Feedbax `AnalysisRunManifest`, records the JSON payload, tracked Markdown/manifest files, and bulk NPZ arrays as an artifact group. |

## Historical Or Provenance-Only Writers

These scripts preserve prior bridge outputs and may be rerun for archaeology,
but they should not be extended with new ad hoc artifact contracts. Port an
active path to `src/rlrmp/analysis/declarative_materialization.py` and
`src/rlrmp/config/analysis_bundles/output_feedback_bridge.yml` first.

| script | status |
|---|---|
| `materialize_output_feedback_gamma_sweep.py` | Historical `97604a8` gamma-sweep sidecar writer. |
| `materialize_output_feedback_lane.py` | Historical `83fc5b5` lane-summary writer. |
| `results/3becdec/scripts/materialize_output_feedback_observer_error_coverage.py` | Historical coverage-sweep writer feeding the `7a459bb` notes. |
| `materialize_output_feedback_sweep_certificates.py` | Legacy active-reference writer for standard certificate rows; math/mode generalization remains on `e6a32b8` before further custody migration. |
| `materialize_output_feedback_failure_decomposition.py` | Legacy companion failure-decomposition writer; keep semantics aligned with standard certificates until migrated. |
| `materialize_output_feedback_interpolated_starts.py` | Historical interpolated-starts bridge writer. |
| `results/1c014e5/scripts/materialize_output_feedback_optimizer_basin_diagnostic.py` | Historical `1c014e5` optimizer-basin writer. |

## Retired Writers

These writers were deleted on `feature/dd8523c-frozen-of-bridge-retirement`.
Recover them from tag `legacy/output-feedback-materializers-retired` only for
archaeology, not as active materializer surfaces.

| script | retired owner |
|---|---|
| `materialize_output_feedback_affine_tracker.py` | `50c260d` |
| `materialize_output_feedback_linear_recurrent.py` | `5e55f69` |
| `materialize_output_feedback_phase_modulated_recurrent.py` | `d6d25d6` |
| `materialize_output_feedback_time_constrained.py` | `87edaae` |
