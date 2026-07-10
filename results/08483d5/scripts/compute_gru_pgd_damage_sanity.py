from __future__ import annotations

# ruff: noqa: E402, F401

import json
from typing import Any, Mapping

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from _common import (
    DamageComputation,
    compute_damage_row as canonical_compute_damage_row,
    damage_payload,
    declared_active_radius,
    delta_summary,
    flattened_per_trial_norm,
    full_qrf_cost,
    full_qrf_cost_context,
    goal_centered_vectors,
    json_ready,
    normalize_per_trial,
    project_per_trial_l2,
    rollout_costs,
    stats,
    summarize_costs,
    summarize_epsilon,
    trial_target_position,
    with_epsilon_delta,
)
from rlrmp.paths import REPO_ROOT

ISSUE = "c92ebd8"
RUN_ID = "moderate_pgd_ofb1p4"
TARGET_DAMAGE = 6131.6906765
N_TRIALS = 64
N_STEPS = 10
SEED = 42
OUTPUT_DIR = REPO_ROOT / "results" / "08483d5" / "notes"
OUTPUT_JSON = OUTPUT_DIR / "gru_pgd_damage_sanity.json"
OUTPUT_MD = OUTPUT_DIR / "gru_pgd_damage_sanity.md"


def main() -> None:
    computation = canonical_compute_damage_row(
        experiment=ISSUE,
        run_id=RUN_ID,
        label="c92ebd8 moderate_pgd_ofb1p4",
        repo_root=REPO_ROOT,
        n_trials=N_TRIALS,
        n_steps=N_STEPS,
        seed=SEED,
    )
    result = build_result(computation)
    OUTPUT_JSON.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(json_ready(result["costs"]), indent=2, sort_keys=True))

def build_result(computation: DamageComputation) -> dict[str, Any]:
    run_spec = computation["run_spec"]
    trial_specs = computation["trial_specs"]
    shared = damage_payload(computation)
    return {
        "schema_version": "rlrmp.08483d5_gru_damage_sanity.v1",
        "selected_model": {
            "issue": ISSUE,
            "run_id": RUN_ID,
            "run_spec_path": str(computation["run"].run_spec_path.relative_to(REPO_ROOT)),
            "artifact_dir": str(computation["run"].artifact_dir.relative_to(REPO_ROOT)),
            "row_rationale": (
                "6D no-integrator H0 no-hold c92 row trained with the gamma 1.4 "
                "output-feedback rollout PGD budget; existing diagnostics show "
                "reach-context attenuation but not a clean across-task robustness win."
            ),
            "checkpoint_policy": "validation_selected_per_replicate",
            "checkpoint_selection": [
                item.to_json(repo_root=REPO_ROOT) for item in computation["checkpoint_selection"]
            ],
        },
        "batch": {
            "n_trials": N_TRIALS,
            "n_replicates": computation["n_replicates"],
            "target_position_m": np.asarray(trial_target_position(trial_specs)).tolist(),
            "initial_position_m": np.asarray(
                trial_specs.inits["mechanics.vector"][..., :2], dtype=np.float64
            ).tolist(),
            "construction": (
                "repeat_single_validation_trial(pair.task.validation_trials, 64), "
                "which repeats the fixed +x 15 cm nominal validation reach."
            ),
            "seed": SEED,
            "paired_rollout_keys": True,
        },
        "adversary": {
            **shared["adversary"],
            "epsilon_dim": int(computation["base_epsilon"].shape[-1]),
            "horizon": int(computation["base_epsilon"].shape[-2]),
            "budget_source": run_spec["hps"]["broad_epsilon_pgd_training"]
            ["budget_contract"]["budget_source"],
        },
        "costs": {
            "lens": "full no-integrator C&S Q/R/Q_f task cost; no disturbance penalty",
            **shared["costs"],
            "target_damage_reference": {
                "source": "Mandible issue 08483d5 first no-launch output-feedback check",
                "paired_nominal_noise_damage": TARGET_DAMAGE,
            },
            "comparison_to_target": {
                "damage_minus_target": computation["damage"]["total"]["mean"] - TARGET_DAMAGE,
                "damage_over_target": computation["damage"]["total"]["mean"] / TARGET_DAMAGE,
            },
        },
        "uncertainties": [
            "This is a small local frozen-batch diagnostic, not a training run.",
            "The adversary is the training-style per-trial direct-epsilon PGD inner loop, not a stored training adversary.",
            "The selected c92 row has empirical reach-context attenuation but existing notes do not claim a clean H-infinity certificate or across-task robustness win.",
        ],
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    costs = result["costs"]
    selected = result["selected_model"]
    adv = result["adversary"]
    comparison = costs["comparison_to_target"]
    return "\n".join(
        [
            "# GRU Damage Sanity Check",
            "",
            "## Selected Row",
            "",
            f"- Issue/run: `{selected['issue']}` / `{selected['run_id']}`.",
            f"- Run spec: `{selected['run_spec_path']}`.",
            f"- Artifact dir: `{selected['artifact_dir']}`.",
            f"- Checkpoint policy: `{selected['checkpoint_policy']}`.",
            f"- Rationale: {selected['row_rationale']}",
            "",
            "Existing tracked evidence: `results/c92ebd8/notes/output_feedback_budget_diagnostics.md` "
            "identifies this as the gamma 1.4 output-feedback-budget PGD row, with "
            "active L2 radius 0.004545011406169036 and reach-context attenuation, "
            "but explicitly says this is not a clean across-task robustness improvement "
            "or formal H-infinity evidence.",
            "",
            "## Frozen-Batch Diagnostic",
            "",
            f"- Batch: {result['batch']['n_trials']} repeated fixed +x 15 cm nominal validation trials "
            f"across {result['batch']['n_replicates']} GRU replicates.",
            f"- Paired rollout seed: `{result['batch']['seed']}`; clean and adversarial costs use the same keys.",
            f"- Adversary: `{adv['optimizer']}` over per-trial direct 6D epsilon sequences, "
            f"{adv['n_steps']} steps, step size {adv['step_size_fraction_of_l2_radius']} x radius.",
            f"- Radius: `{adv['l2_radius_per_trial']:.16g}` per trial; selected epsilon mean norm "
            f"`{adv['selected_epsilon']['l2_norm_mean_per_trial']:.8g}`, max norm "
            f"`{adv['selected_epsilon']['l2_norm_max_per_trial']:.8g}`; boundary fraction "
            f"`{adv['selected_epsilon']['boundary_fraction']:.3f}`.",
            f"- Selected adversarial delta energy: mean per trial "
            f"`{adv['selected_epsilon']['energy_mean_per_trial']:.8g}`, total across batch "
            f"`{adv['selected_epsilon']['energy_total']:.8g}`.",
            f"- Base nominal epsilon mean norm: `{adv['base_epsilon']['l2_norm_mean_per_trial']:.8g}`.",
            "",
            "## Costs",
            "",
            "| quantity | clean | adversarial | paired damage |",
            "|---|---:|---:|---:|",
            f"| total | {costs['clean']['total']['mean']:.8g} | "
            f"{costs['adversarial']['total']['mean']:.8g} | "
            f"{costs['paired_damage']['total']['mean']:.8g} |",
            f"| stage state | {costs['clean']['stage_state']['mean']:.8g} | "
            f"{costs['adversarial']['stage_state']['mean']:.8g} | "
            f"{costs['paired_damage']['stage_state']['mean']:.8g} |",
            f"| control | {costs['clean']['control']['mean']:.8g} | "
            f"{costs['adversarial']['control']['mean']:.8g} | "
            f"{costs['paired_damage']['control']['mean']:.8g} |",
            f"| terminal | {costs['clean']['terminal']['mean']:.8g} | "
            f"{costs['adversarial']['terminal']['mean']:.8g} | "
            f"{costs['paired_damage']['terminal']['mean']:.8g} |",
            "",
            "Costs are full no-integrator C&S Q/R/Q_f task costs; the disturbance penalty is not subtracted.",
            "",
            "## Comparison To 08483d5 Target",
            "",
            f"- Reference paired-noise output-feedback damage: `{TARGET_DAMAGE:.8g}`.",
            f"- GRU paired damage / reference: `{comparison['damage_over_target']:.4g}`.",
            f"- Difference: `{comparison['damage_minus_target']:.8g}`.",
            "",
            "Interpretation: same order of magnitude if the ratio is near 1; clearly smaller if far below 1.",
            "",
            "## Uncertainties",
            "",
            *[f"- {item}" for item in result["uncertainties"]],
            "",
            "## Machine Output",
            "",
            f"- JSON: `{OUTPUT_JSON}`",
            f"- Script: `{OUTPUT_DIR / 'gru_damage_sanity.py'}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()
