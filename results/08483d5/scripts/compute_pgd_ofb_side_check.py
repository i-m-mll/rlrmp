from __future__ import annotations

# ruff: noqa: E402

import json
from functools import partial
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from _common import (
    DamageComputation,
    compute_damage_row as canonical_compute_damage_row,
    damage_payload,
    json_ready,
)
from rlrmp.paths import portable_repo_path

from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    evaluate_stochastic_forward_velocity_profile,
    resolve_run_inputs,
)
from rlrmp.io import update_marked_section


REPO_ROOT = Path(__file__).resolve().parents[3]
repo_rel = partial(portable_repo_path, repo_root=REPO_ROOT)
OUTPUT_JSON = REPO_ROOT / "results" / "08483d5" / "notes" / "pgd_ofb_side_check.json"
OUTPUT_MD = REPO_ROOT / "results" / "08483d5" / "notes" / "pgd_ofb_side_check.md"
TARGET_DAMAGE = 6131.6906765
N_TRIALS = 64
N_STEPS = 10
SEED = 42

ROWS = (
    {
        "label": "c92 6D no-PGD moderate",
        "experiment": "c92ebd8",
        "run_id": "open_loop_moderate",
        "role": "baseline_6d_no_pgd",
        "compute_damage": False,
    },
    {
        "label": "c92 moderate PGD OFB gamma 1.05",
        "experiment": "c92ebd8",
        "run_id": "moderate_pgd_ofb1p05",
        "role": "older_pgd_row",
        "compute_damage": True,
    },
    {
        "label": "c92 moderate PGD OFB gamma 1.4",
        "experiment": "c92ebd8",
        "run_id": "moderate_pgd_ofb1p4",
        "role": "older_pgd_row",
        "compute_damage": True,
    },
    {
        "label": "33b0dcb H0 const_band16 no-PGD",
        "experiment": "33b0dcb",
        "run_id": "h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64",
        "role": "baseline_8d_context",
        "compute_damage": False,
    },
)


def main() -> None:
    """Compute the older-PGD side check requested under issue 08483d5."""

    velocity_rows = [compute_velocity_row(row) for row in ROWS]
    baseline = next(row for row in velocity_rows if row["role"] == "baseline_6d_no_pgd")
    for row in velocity_rows:
        row["relative_to_6d_baseline"] = velocity_relative_to(row, baseline)

    damage_rows = [compute_damage_row(row) for row in ROWS if bool(row["compute_damage"])]
    output = {
        "schema_version": "rlrmp.08483d5_pgd_ofb_side_check.v1",
        "issue": "08483d5",
        "created_by": "results/08483d5/scripts/compute_pgd_ofb_side_check.py",
        "batch_contract": {
            "n_trials": N_TRIALS,
            "construction": (
                "repeat_single_validation_trial(pair.task.validation_trials, 64), "
                "the fixed +x nominal 15 cm validation reach"
            ),
            "rollout_key_seed": SEED,
            "paired_clean_and_adversarial_keys": True,
        },
        "checkpoint_policy": (
            "validation_selected_per_replicate using sparse history, matching the prior "
            "moderate_pgd_ofb1p4 sanity check"
        ),
        "noise_convention": (
            "GRU stochastic rollout noise from Task.eval_trials with identical PRNG keys "
            "for paired clean/adversarial damage; peak velocity uses the same repeated "
            "nominal validation trial convention pooled across replicates and stochastic trials"
        ),
        "cost_definition": (
            "full no-integrator C&S Q/R/Q_f task cost for 6D PGD rows; disturbance "
            "penalty is not subtracted"
        ),
        "damage_target_reference": {
            "source": "08483d5 output-feedback paired nominal-noise damage check",
            "paired_nominal_noise_damage": TARGET_DAMAGE,
        },
        "velocity_rows": velocity_rows,
        "damage_rows": damage_rows,
        "existing_evidence_paths": {
            "prior_ofb1p4_damage": "results/08483d5/notes/gru_pgd_damage_sanity.json",
            "c92_ofb_budget_diagnostics": "results/c92ebd8/notes/output_feedback_budget_diagnostics.md",
            "c92_ofb_budget_diagnostics_json": "results/c92ebd8/notes/output_feedback_budget_diagnostics.json",
            "c92_final_checkpoint_velocity_figure_spec": (
                "results/c92ebd8/figures/"
                "pgd_ofb_budget_moderate_nominal_velocity_profiles/spec.json"
            ),
        },
        "uncertainties": [
            "The c92 rows are 6D no-integrator GRUs; the 33b0dcb const_band16 baseline is 8D and is reported only as baseline-context provenance.",
            "The adversaries here were recomputed by a local 10-step projected-gradient ascent on the frozen nominal batch; they are not stored training adversaries.",
            "Both PGD damage estimates are small frozen-batch diagnostics, not new training runs or launch recommendations.",
            "The historical OFB radii are reported only as provenance for these older rows, not as recommended defaults.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(output), indent=2, sort_keys=True) + "\n")
    update_marked_section(OUTPUT_MD, "pgd_ofb_side_check", render_markdown(output))
    print(json.dumps(summary_for_stdout(output), indent=2, sort_keys=True))


def compute_velocity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    run = resolve_run_inputs(
        experiment=str(row["experiment"]),
        run_ids=(str(row["run_id"]),),
        labels=(str(row["label"]),),
        repo_root=REPO_ROOT,
    )[0]
    profile = evaluate_stochastic_forward_velocity_profile(
        run,
        experiment=str(row["experiment"]),
        n_rollout_trials=DEFAULT_N_ROLLOUT_TRIALS,
        use_validation_selected_checkpoints=True,
        repo_root=REPO_ROOT,
    )
    peak_idx = int(np.argmax(profile.mean))
    run_spec = run.run_spec
    return {
        "label": row["label"],
        "role": row["role"],
        "experiment": row["experiment"],
        "run_id": row["run_id"],
        "run_spec_path": repo_rel(run.run_spec_path),
        "artifact_dir": repo_rel(run.artifact_dir),
        "model_identity": model_identity(run_spec),
        "checkpoint_policy": "validation_selected_per_replicate",
        "checkpoint_selection": checkpoint_selection_json(profile.checkpoint_selection),
        "n_replicates": int(profile.n_replicates),
        "n_rollout_trials_per_replicate": int(profile.n_rollout_trials_per_replicate),
        "peak_forward_velocity_m_s": float(profile.mean[peak_idx]),
        "time_of_peak_forward_velocity_s": float(profile.time_s[peak_idx]),
        "replicate_peak_mean_m_s": stats(np.max(profile.replicate_mean, axis=1)),
        "mean_profile_source": "peak of pooled mean forward velocity profile",
    }


def compute_damage_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compute and format one PGD side-check damage row."""

    computation = canonical_compute_damage_row(
        experiment=str(row["experiment"]),
        run_id=str(row["run_id"]),
        label=str(row["label"]),
        repo_root=REPO_ROOT,
        n_trials=N_TRIALS,
        n_steps=N_STEPS,
        seed=SEED,
    )
    return format_damage_row(row, computation)


def format_damage_row(
    row: Mapping[str, Any], computation: DamageComputation
) -> dict[str, Any]:
    run_spec = computation["run_spec"]
    contract = run_spec["hps"]["broad_epsilon_pgd_training"]["budget_contract"]
    shared = damage_payload(computation)
    return {
        "label": row["label"],
        "experiment": row["experiment"],
        "run_id": row["run_id"],
        "run_spec_path": repo_rel(computation["run"].run_spec_path),
        "artifact_dir": repo_rel(computation["run"].artifact_dir),
        "model_identity": model_identity(run_spec),
        "checkpoint_policy": "validation_selected_per_replicate",
        "checkpoint_selection": checkpoint_selection_json(computation["checkpoint_selection"]),
        "adversary_recomputed": True,
        "adversary": {
            **shared["adversary"],
            "budget_contract": {
                "active_max_l2_radius_15cm": float(contract["active_max_l2_radius_15cm"]),
                "effective_l2_radius_15cm": float(contract["effective_l2_radius_15cm"]),
                "gamma_factor": float(contract["gamma_factor"]),
                "budget_source": contract.get("budget_source"),
            },
        },
        "costs": shared["costs"]
        | {"damage_over_target": float(computation["damage"]["total"]["mean"] / TARGET_DAMAGE)},
    }


def model_identity(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    hps = run_spec["hps"]
    model = hps["model"]
    pgd = hps.get("broad_epsilon_pgd_training", {})
    return {
        "hidden_type": hps.get("hidden_type"),
        "hidden_size": int(model["hidden_size"]),
        "n_replicates": int(model["n_replicates"]),
        "initial_hidden_encoder": bool(model.get("initial_hidden_encoder", False)),
        "physical_state_dim": int(model["physical_state_dim"]),
        "state_dim": int(model["state_dim"]),
        "no_integrator_state": bool(model.get("no_integrator_state", False)),
        "loss_objective": hps["loss"]["objective"],
        "broad_epsilon_pgd_enabled": bool(pgd.get("enabled", False)),
        "broad_epsilon_pgd_level": pgd.get("level"),
        "broad_epsilon_pgd_mode": pgd.get("mode"),
        "target_distance_m": run_spec.get("game_card", {}).get("target_distance_m"),
    }


def checkpoint_selection_json(selections: Sequence[Any]) -> list[dict[str, Any]]:
    out = []
    for selection in selections:
        if hasattr(selection, "to_json"):
            out.append(selection.to_json(repo_root=REPO_ROOT))
        else:
            out.append(json_ready(selection))
    return out


def velocity_relative_to(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    value = float(row["peak_forward_velocity_m_s"])
    ref = float(baseline["peak_forward_velocity_m_s"])
    return {
        "delta_m_s": value - ref,
        "ratio": value / ref,
        "percent_delta": 100.0 * (value - ref) / ref,
    }


def stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    velocity_rows = result["velocity_rows"]
    damage_rows = result["damage_rows"]
    lines = [
        "# PGD OFB Side Check",
        "",
        "## Scope",
        "",
        "This local side check answers the older-PGD questions for issue `08483d5`. "
        "It does not launch training, update controller weights, request auth, push, "
        "or treat historical cap/radius/trust-region values as new defaults.",
        "",
        "Checkpoint policy: validation-selected per replicate using sparse history. "
        "Batch convention: 64 repeats of the fixed +x nominal 15 cm validation reach. "
        "Noise convention: paired clean/adversarial damage uses identical stochastic "
        "rollout keys; nominal peak velocity pools repeated stochastic validation "
        "trials across replicates.",
        "",
        "## Nominal Peak Velocity",
        "",
        "| row | dim | peak m/s | delta vs c92 no-PGD | percent | run spec |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in velocity_rows:
        rel = row["relative_to_6d_baseline"]
        lines.append(
            f"| `{row['run_id']}` | {row['model_identity']['physical_state_dim']}D | "
            f"{row['peak_forward_velocity_m_s']:.6f} | {rel['delta_m_s']:+.6f} | "
            f"{rel['percent_delta']:+.2f}% | `{row['run_spec_path']}` |"
        )
    lines.extend(
        [
            "",
            "The 6D c92 no-PGD baseline is the direct same-family baseline for the two older "
            "PGD rows. The 33b0dcb const_band16 row is shown as live baseline context, "
            "but it is 8D and therefore not the dimensional match for the c92 PGD rows.",
            "",
            "## Recomputed PGD Damage",
            "",
            "| row | radius | gamma factor | clean cost | adversarial cost | paired damage | damage/reference | boundary fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in damage_rows:
        adv = row["adversary"]
        costs = row["costs"]
        contract = adv["budget_contract"]
        lines.append(
            f"| `{row['run_id']}` | {adv['l2_radius_per_trial']:.10g} | "
            f"{contract['gamma_factor']:.3g} | {costs['clean']['total']['mean']:.4f} | "
            f"{costs['adversarial']['total']['mean']:.4f} | "
            f"{costs['paired_damage']['total']['mean']:.4f} | "
            f"{costs['damage_over_target']:.4f} | "
            f"{adv['selected_epsilon']['boundary_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Cost definition: full no-integrator C&S Q/R/Q_f task cost for the 6D PGD rows; "
            "the disturbance penalty is not subtracted. The reference damage is the "
            "paired nominal-noise output-feedback damage from the first 08483d5 check "
            f"({TARGET_DAMAGE:.4f}).",
            "",
            "## Provenance",
            "",
            "- `moderate_pgd_ofb1p05` uses the historical output-feedback rollout radius "
            "`ofb_6d_no_integrator_gamma_1p05_rollout_radius`; this is provenance only.",
            "- `moderate_pgd_ofb1p4` uses the historical output-feedback rollout radius "
            "`ofb_6d_no_integrator_gamma_1p4_rollout_radius`; this is provenance only.",
            "- Existing prior `ofb1p4` damage path: "
            "`results/08483d5/notes/gru_pgd_damage_sanity.json`.",
            "- Existing c92 validation-selected diagnostic table: "
            "`results/c92ebd8/notes/output_feedback_budget_diagnostics.md`.",
            "",
            "## Uncertainty",
            "",
            *[f"- {item}" for item in result["uncertainties"]],
            "",
            "## Outputs",
            "",
            f"- JSON sidecar: `{repo_rel(OUTPUT_JSON)}`",
            "- Script: `results/08483d5/scripts/compute_pgd_ofb_side_check.py`",
            "",
        ]
    )
    return "\n".join(lines)


def summary_for_stdout(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "velocity": [
            {
                "run_id": row["run_id"],
                "peak_forward_velocity_m_s": row["peak_forward_velocity_m_s"],
                "delta_vs_6d_baseline_m_s": row["relative_to_6d_baseline"]["delta_m_s"],
            }
            for row in result["velocity_rows"]
        ],
        "damage": [
            {
                "run_id": row["run_id"],
                "clean": row["costs"]["clean"]["total"]["mean"],
                "adversarial": row["costs"]["adversarial"]["total"]["mean"],
                "paired_damage": row["costs"]["paired_damage"]["total"]["mean"],
                "damage_over_target": row["costs"]["damage_over_target"],
                "boundary_fraction": row["adversary"]["selected_epsilon"]["boundary_fraction"],
            }
            for row in result["damage_rows"]
        ],
        "outputs": {
            "json": repo_rel(OUTPUT_JSON),
            "markdown": repo_rel(OUTPUT_MD),
        },
    }


if __name__ == "__main__":
    main()
