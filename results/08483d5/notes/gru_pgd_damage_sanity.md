# GRU Damage Sanity Check

## Selected Row

- Issue/run: `c92ebd8` / `moderate_pgd_ofb1p4`.
- Run spec: `results/c92ebd8/runs/moderate_pgd_ofb1p4.json`.
- Artifact dir: `_artifacts/c92ebd8/runs/moderate_pgd_ofb1p4`.
- Checkpoint policy: `validation_selected_per_replicate`.
- Rationale: 6D no-integrator H0 no-hold c92 row trained with the gamma 1.4 output-feedback rollout PGD budget; existing diagnostics show reach-context attenuation but not a clean across-task robustness win.

Existing tracked evidence: `results/c92ebd8/notes/output_feedback_budget_diagnostics.md` identifies this as the gamma 1.4 output-feedback-budget PGD row, with active L2 radius 0.004545011406169036 and reach-context attenuation, but explicitly says this is not a clean across-task robustness improvement or formal H-infinity evidence.

## Frozen-Batch Diagnostic

- Batch: 64 repeated fixed +x 15 cm nominal validation trials across 5 GRU replicates.
- Paired rollout seed: `42`; clean and adversarial costs use the same keys.
- Adversary: `projected_gradient_ascent` over per-trial direct 6D epsilon sequences, 10 steps, step size 0.25 x radius.
- Radius: `0.004545011586771563` per trial; selected epsilon mean norm `0.0045450116`, max norm `0.0045450116`; boundary fraction `1.000`.
- Selected adversarial delta energy: mean per trial `2.065713e-05`, total across batch `0.0013220563`.
- Base nominal epsilon mean norm: `0.05530405`.

## Costs

| quantity | clean | adversarial | paired damage |
|---|---:|---:|---:|
| total | 4683.2999 | 8180.753 | 3497.4532 |
| stage state | 2761.6984 | 6058.9764 | 3297.2781 |
| control | 1915.0921 | 1843.028 | -72.064065 |
| terminal | 6.5094229 | 278.74859 | 272.23916 |

Costs are full no-integrator C&S Q/R/Q_f task costs; the disturbance penalty is not subtracted.

## Comparison To 08483d5 Target

- Reference paired-noise output-feedback damage: `6131.6907`.
- GRU paired damage / reference: `0.5704`.
- Difference: `-2634.2375`.

Interpretation: same order of magnitude if the ratio is near 1; clearly smaller if far below 1.

## Uncertainties

- This is a small local frozen-batch diagnostic, not a training run.
- The adversary is the training-style per-trial direct-epsilon PGD inner loop, not a stored training adversary.
- The selected c92 row has empirical reach-context attenuation but existing notes do not claim a clean H-infinity certificate or across-task robustness win.

## Machine Output

- JSON: `results/08483d5/notes/gru_pgd_damage_sanity.json`
- Script: `results/08483d5/scripts/compute_gru_pgd_damage_sanity.py`
