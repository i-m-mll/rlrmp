# c92 artifact organization

This note records the c92 artifact sorting pass for the calibrated perturbation
matrix post-run outputs.

## Bulk artifact layout

The c92 bulk tree now keeps analysis payloads under analysis-family directories:

- `_artifacts/c92ebd8/runs/`: model checkpoints, training histories, summaries,
  and diagnostics for the run variants.
- `_artifacts/c92ebd8/figures/`: HTML figure renders and the existing PNG
  exports.
- `_artifacts/c92ebd8/evaluation_diagnostics/`: evaluation diagnostic NPZ
  outputs.
- `_artifacts/c92ebd8/perturbation_response/`: perturbation-response caches,
  including the validation-selected GRU cache and the PGD 1.05 reach/profile
  diagnostic caches moved in this pass.
- `_artifacts/c92ebd8/stabilization_diagnostics/`: stabilization-task
  diagnostic detail payloads, including the PGD robustness isolation packet.
- `_artifacts/c92ebd8/runpod/`: local provider/run provenance payloads.

Before this pass, the following analysis payloads were top-level children of
`_artifacts/c92ebd8/`:

- `pgd_1p05_moderate_perturbation_profiles_overlay/`
- `pgd_1p05_reach_context_diagnostics/`
- the legacy-named top-level stabilization-task diagnostics directory

They were moved to:

- `_artifacts/c92ebd8/perturbation_response/pgd_1p05_moderate_perturbation_profiles_overlay/`
- `_artifacts/c92ebd8/perturbation_response/pgd_1p05_reach_context_diagnostics/`
- `_artifacts/c92ebd8/stabilization_diagnostics/pgd_1p05_stabilization_diagnostics/`
- `_artifacts/c92ebd8/stabilization_diagnostics/pgd_robustness_isolation/`

The stale top-level stabilization-task directory was not kept. Its
`per_probe_detail.json` was the fuller payload: the previously nested copy had
the same rows and scalar probe metrics but omitted trajectory/profile fields.
The fuller payload is now the canonical local detail file at
`_artifacts/c92ebd8/stabilization_diagnostics/pgd_1p05_stabilization_diagnostics/per_probe_detail.json`.

The PGD robustness isolation packet is mixed, but its bulk outputs are
stabilization-response detail plus matched-reach sidecars, so it is kept under
`stabilization_diagnostics/pgd_robustness_isolation/` instead of becoming a new
top-level per-analysis directory.

Path references in the local c92 figure spec, reach-context note, generator, and
moved detail manifests were updated to match.

## Naming compatibility

Durable names now use `stabilization` or `stabilization_task` for this
diagnostic family. The historical `steady_state_hold` phrase may still appear in
older Mandible comments or captured regeneration-spec git-status snapshots, but
current c92 notes/specs/manifests should not point at the removed top-level
stabilization-task diagnostics directory.

## Remaining local-only items

- `_artifacts/c92ebd8/plotly_example.json` remains at the c92 artifact root.
  No tracked c92 note/spec references it; treat it as an unsorted local-only
  scratch file unless a future owner identifies a durable role for it.
- `.DS_Store` files remain local filesystem metadata and were not cleaned up in
  this pass.

## PNG export status

Current c92 figure-family status after the 2026-06-28 PNG backfill:

| Source figure family | HTML renders | PNG renders | PNG output directory | Status |
|---|---:|---:|---|---|
| `gru_postrun_validation_selected_moderate` | 5 | 5 | `_artifacts/c92ebd8/figures/gru_postrun_validation_selected_moderate_pngs/` | backfilled |
| `moderate_perturbation_profiles` | 288 | 288 | `_artifacts/c92ebd8/figures/moderate_perturbation_profiles_pngs/` | backfilled |
| `moderate_perturbation_residual_pngs` | 0 | 144 | `_artifacts/c92ebd8/figures/moderate_perturbation_residual_pngs/` | pre-existing PNG-only residual subset |
| `nominal_velocity_profiles` | 1 | 1 | `_artifacts/c92ebd8/figures/nominal_velocity_profiles_pngs/` | backfilled |
| `pgd_1p05_moderate_perturbation_profiles_overlay` | 96 | 96 | `_artifacts/c92ebd8/figures/pgd_1p05_moderate_perturbation_profiles_overlay_pngs/` | backfilled |
| `pgd_1p05_moderate_perturbation_response_overlays` | 1 | 1 | `_artifacts/c92ebd8/figures/pgd_1p05_moderate_perturbation_response_overlays_pngs/` | backfilled |
| `pgd_1p05_nominal_velocity_profiles` | 1 | 1 | `_artifacts/c92ebd8/figures/pgd_1p05_nominal_velocity_profiles_pngs/` | backfilled |
| `pgd_1p05_stabilization_perturbation_responses` | 5 | 5 | `_artifacts/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/` | completed by stabilization lane |

The backfill used Playwright screenshots of the generated Plotly HTML files.
The PNG directories preserve each source family's nested row/channel layout.
No new diagnostics, training runs, or artifact reorganization were part of this
backfill.
