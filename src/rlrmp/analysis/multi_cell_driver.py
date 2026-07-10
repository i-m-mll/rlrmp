"""Shared orchestration primitives for multi-cell reach analyses."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.plot import save_figure

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.io import update_marked_section


_TRAINING_ARG_PROFILES: dict[str, dict[str, Any]] = {
    "lit_replication": {
        "n_adversary_batches": 0,
        "batch_size": 250,
        "nn_output_jerk": 0.0,
        "seed": 42,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "loss_update_enabled": False,
        "loss_update_ratio": 0.5,
        "effector_pos_running": 1.0,
        "effector_hold_pos": 1.0,
        "effector_hold_vel": 0.0,
        "effector_pos_late_weight": 0.0,
        "effector_vel_late": 0.0,
        "effector_final_vel": 0.0,
        "effector_pos_late_final_scale": 2.0,
        "effector_pos_late_start_step": 80,
        "p_catch_trial": 0.5,
        "nn_output": 1e-5,
        "nn_hidden": 1e-5,
        "nn_hidden_derivative": 0.001,
        "nn_output_pre_go": 0.0,
        "nn_hidden_derivative_pre_go": 0.0,
        "effector_pos_running_schedule": "flat",
        "effector_hold_pos_schedule": "flat",
        "position_powerlaw_power": 6.0,
        "controller_lr": 1e-4,
    },
    "movement_ramp": {
        "n_adversary_batches": 0,
        "batch_size": 250,
        "seed": 42,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "loss_update_enabled": False,
        "loss_update_ratio": 0.5,
        "effector_pos_running": 1.0,
        "effector_hold_pos": 0.0,
        "effector_hold_vel": 0.0,
        "effector_pos_late_weight": 0.0,
        "effector_vel_late": 0.0,
        "effector_final_vel": 0.0,
        "effector_pos_late_final_scale": 2.0,
        "effector_pos_late_start_step": 80,
        "p_catch_trial": 0.5,
        "nn_output": 1e-5,
        "nn_hidden": 1e-5,
        "nn_output_jerk": 0.0,
        "nn_hidden_derivative": 0.001,
        "nn_hidden_derivative_pre_go": 0.0,
        "effector_pos_running_schedule": "movement_ramp",
        "effector_hold_pos_schedule": "flat",
        "position_powerlaw_power": 6.0,
        "controller_lr": 1e-4,
    },
    "anti_anticipation": {
        "n_adversary_batches": 0,
        "batch_size": 250,
        "nn_output_jerk": 1e5,
        "seed": 42,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "loss_update_enabled": False,
        "loss_update_ratio": 0.5,
        "effector_pos_running": 1.0,
        "effector_pos_late_weight": 0.5,
        "effector_vel_late": 0.1,
        "effector_final_vel": 0.0,
        "effector_pos_late_final_scale": 2.0,
        "effector_pos_late_start_step": 80,
        "nn_hidden_derivative": 0.0,
        "nn_output_pre_go": 0.0,
        "nn_hidden_derivative_pre_go": 0.0,
        "controller_lr": 1e-4,
    },
}


def args_namespace(
    defaults: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    profile: str | None = None,
    n_warmup_batches: int | None = None,
    n_replicates: int | None = None,
) -> argparse.Namespace:
    """Build a CLI-compatible namespace from shared defaults and row overrides."""

    if profile is not None and defaults is not None:
        raise ValueError("pass defaults or profile, not both")
    values = dict(_TRAINING_ARG_PROFILES[profile]) if profile is not None else dict(defaults or {})
    if n_warmup_batches is not None:
        values["n_warmup_batches"] = n_warmup_batches
    if n_replicates is not None:
        values["n_replicates"] = n_replicates
    values.update(overrides or {})
    return argparse.Namespace(**values)


def legacy_task_trainer_history_skeleton(*args: Any, **kwargs: Any) -> Any:
    """Build a producing-era TaskTrainer history skeleton or fail clearly.

    These result-only readers deserialize the removed fixed-array
    ``TaskTrainerHistory`` format. Current Feedbax exposes event history instead and has no
    public equivalent for the old Equinox tree shape, so executing these historical readers
    requires their producing Feedbax checkout.
    """

    try:
        legacy_train = importlib.import_module("feedbax.train")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "historical TaskTrainerHistory reader requires the producing Feedbax checkout; "
            "current Feedbax has no compatible fixed-array history API"
        ) from error
    return legacy_train.init_task_trainer_history(*args, **kwargs)


def compute_kinematics_per_replicate(
    states: Any,
    trial_specs: Any,
    *,
    pre_go_window_steps: int = 20,
) -> dict[str, np.ndarray]:
    """Compute per-replicate forward-kinematic and pre-go drift metrics."""

    pos = states.mechanics.effector.pos
    vel = states.mechanics.effector.vel
    target_key = list(trial_specs.targets.keys())[0]
    goal = trial_specs.targets[target_key].value[:, -1, :]
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]
    _n_rep, _n_trials, n_steps, _dims = pos.shape
    time = jnp.arange(n_steps)
    after_go = time[None, None, :] >= go_idx[None, :, None]
    before_go = time[None, None, :] < go_idx[None, :, None]
    pre_go_window = before_go & (
        time[None, None, :] >= go_idx[None, :, None] - pre_go_window_steps
    )

    def positions_at_go(pos_rep: Any, indices: Any) -> Any:
        return jax.vmap(lambda trial, index: trial[index])(pos_rep, indices)

    pos_at_go = jax.vmap(positions_at_go, in_axes=(0, None))(pos, go_idx)
    direction = goal[None, :, :] - pos_at_go
    direction /= jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    forward_velocity = jnp.sum(vel * direction[:, :, None, :], axis=-1)
    post_go_velocity = jnp.where(after_go, forward_velocity, 0.0)
    peak_forward = jnp.max(post_go_velocity, axis=-1)
    time_to_peak = jnp.maximum(jnp.argmax(post_go_velocity, axis=-1) - go_idx[None, :], 0)
    relative_position = pos - pos[:, :, :1, :]
    forward_position = jnp.sum(relative_position * direction[:, :, None, :], axis=-1)
    hold_drift = jnp.where(before_go, forward_position, -jnp.inf).max(axis=-1)
    hold_drift = jnp.where(jnp.isinf(hold_drift), 0.0, hold_drift) * 1000.0
    masked = jnp.where(pre_go_window, forward_position, 0.0)
    counts = jnp.maximum(jnp.sum(pre_go_window, axis=-1), 1)
    return {
        "peak_forward_velocity": np.asarray(peak_forward),
        "time_to_peak_after_go": np.asarray(time_to_peak),
        "forward_vel_profile": np.asarray(forward_velocity),
        "pos_forward_profile": np.asarray(forward_position),
        "hold_drift_mm": np.asarray(hold_drift),
        "pre_go_rms_mm": np.asarray(jnp.sqrt(jnp.sum(masked**2, axis=-1) / counts) * 1000.0),
        "pre_go_mean_mm": np.asarray(jnp.sum(masked, axis=-1) / counts * 1000.0),
        "go_idx": np.asarray(go_idx),
    }


def run_replicate_kinematics_analysis(
    hooks: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    """Run the shared six-cell evaluation, figure, table, and note workflow."""

    if profile not in {"lit_replication", "anti_anticipation"}:
        raise ValueError(f"unknown multi-cell analysis profile: {profile}")
    lit = profile == "lit_replication"
    repo_root = Path(hooks["REPO_ROOT"])
    experiment = str(hooks["EXPERIMENT"])
    labels = tuple(hooks["CELL_LABELS"])
    display_names = hooks["CELL_DISPLAY_NAMES"]
    n_replicates = int(hooks["N_REPLICATES"])

    parser = argparse.ArgumentParser(description=hooks.get("__doc__"))
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=None,
        help="Base directory containing the 6 cell subdirs with adversarial_model.eqx",
    )
    parser.add_argument(
        "--sisu",
        type=float,
        default=0.5,
        help="SISU level for evaluation (default: 0.5)",
    )
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args()
    artifact_base = args.artifact_base or (
        repo_root / "_artifacts" / experiment / "runs"
        if lit
        else repo_root / "_artifacts" / "2bc95fd"
    )
    results_base = repo_root / "results" / (experiment if lit else "2bc95fd")
    notes_dir = results_base / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    if not lit:
        for figure_name in (
            "peak_velocity_distributions",
            "forward_velocity_profiles",
            "hold_drift_profiles",
            "rmse_ratio_comparison",
        ):
            (artifact_base / "figures" / figure_name).mkdir(parents=True, exist_ok=True)
            (results_base / "figures" / figure_name).mkdir(parents=True, exist_ok=True)

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")
    cell_stats: dict[str, dict[str, Any]] = {}
    cell_kms: dict[str, dict[str, Any]] = {}
    input_artifacts: list[dict[str, Any]] = []
    for label in labels:
        print(f"\n[{label}] Loading model ...", flush=True)
        try:
            model, task, n_reps = hooks["load_cell_model"](label, artifact_base)
        except FileNotFoundError as error:
            print(f"  SKIP: {error}")
            continue
        except Exception as error:
            import traceback

            print(f"  FAILED loading: {type(error).__name__}: {error}")
            traceback.print_exc()
            continue
        print(f"  Loaded. n_replicates={n_reps}")
        checkpoint = artifact_base / label / "adversarial_model.eqx"
        input_artifacts.append({"path": str(checkpoint), "role": f"adversarial_model:{label}"})
        try:
            trial_specs = hooks["build_zero_pert_trials"](task, sisu=args.sisu)
            n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
            print(f"  Evaluating {n_reps} replicates on {n_trials} trials ...", flush=True)
            states = hooks["eval_ensemble"](
                task,
                model,
                trial_specs,
                key=jr.PRNGKey(args.eval_seed),
                n_replicates=n_reps,
            )
            print("  Eval OK. Computing kinematics ...", flush=True)
            kinematics = compute_kinematics_per_replicate(states, trial_specs)
            stats = hooks["compute_cell_stats"](kinematics)
            cell_kms[label] = kinematics
            cell_stats[label] = stats
        except Exception as error:
            import traceback

            print(f"  FAILED eval/kinematics: {type(error).__name__}: {error}")
            traceback.print_exc()
            continue
        plus_minus = "+/-" if lit else "±"
        print(
            f"  peak_vel: {stats['mean_peak_velocity']:.4f} {plus_minus} "
            f"{stats['sd_peak_velocity']:.4f} m/s  CV={stats['cv_peak_vel']:.3f}  "
            f"hold_drift={stats['mean_hold_drift_mm']:.2f} {plus_minus} "
            f"{stats['sd_hold_drift_mm']:.2f} mm"
        )
    if not cell_stats:
        print("\nNo cells loaded -- aborting." if lit else "\nNo cells loaded — aborting.")
        return

    print("\n--- Computing pairwise RMSE ratios (primary metric) ---")
    rmse_ratios: dict[str, dict[str, Any]] = {}
    if len(cell_kms) >= 2:
        rmse_ratios = hooks["compute_rmse_ratios"](cell_kms)
        for label in labels:
            if label not in rmse_ratios:
                continue
            row = rmse_ratios[label]
            print(
                f"  [{label}] vel-RMSE-ratio={row['vel_rmse_ratio']:.3f}  "
                f"(within={row['vel_within_rmse']:.4f}, "
                f"nearest-across={row['vel_nearest_across_rmse']:.4f})  "
                f"pos-RMSE-ratio={row['pos_rmse_ratio']:.3f}"
            )
    else:
        print("  Less than 2 cells loaded -- cannot compute cross-cell RMSE ratios.")

    figure_experiment = experiment if lit else "anti_anticipation_loss_shape_6cell"
    route_experiment = experiment if lit else "2bc95fd"
    shared_plot = {
        "cells": labels,
        "n_replicates": n_replicates,
        "sisu": args.sisu,
        "pert_scale": 0.0,
    }
    shared_eval = {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}}
    print("\n--- Building figures ---")
    peak_spec = {
        "figure_kind": "peak_velocity_distributions_violin",
        "experiment": figure_experiment,
        "inputs": input_artifacts,
        "transform": [
            shared_eval,
            {"name": "compute_peak_forward_velocity", "kwargs": {}},
            {"name": "mean_over_trials_per_replicate", "kwargs": {}},
        ],
        "plot_kwargs": shared_plot,
        "metric_note": (
            "Annotation shows CV (SD/mean of peak vel scalar) -- auxiliary metric. "
            if lit
            else "Annotation shows CV (SD/mean of peak vel scalar) — auxiliary metric. "
        )
        + "Primary metric is vel_rmse_ratio in rmse_ratio_comparison figure.",
        "cell_stats": {
            label: {
                key: stats[key]
                for key in (
                    "mean_peak_velocity",
                    "sd_peak_velocity",
                    "cv_peak_vel",
                    "peak_vel_per_rep",
                )
            }
            for label, stats in cell_stats.items()
        },
    }
    _save_multi_cell_figure(
        hooks["make_peak_velocity_figure"](cell_stats),
        peak_spec,
        experiment=route_experiment,
        topic="peak_velocity_distributions",
    )
    if cell_kms:
        profile_transforms = [
            shared_eval,
            {"name": "forward_velocity_projection_onto_reach_axis", "kwargs": {}},
            {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
            {"name": "trim_to_full_support", "kwargs": {"min_coverage": 1.0}},
            {"name": "replicate_nanmean_over_trials", "kwargs": {}},
        ]
        profile_plot = {
            **shared_plot,
            "dt": 0.01,
            "alignment": "go_cue_per_trial",
            "shared_yaxes": "all",
        }
        _save_multi_cell_figure(
            hooks["make_forward_velocity_profile_figure"](cell_kms),
            {
                "figure_kind": "forward_velocity_profile_time_series_go_aligned",
                "experiment": figure_experiment,
                "inputs": input_artifacts,
                "transform": profile_transforms,
                "plot_kwargs": profile_plot,
                "fix_note": (
                    "Bug: 06f7faf — go-cue alignment + trim-to-full-support + shared "
                    "y-axes across cells."
                ),
            },
            experiment=route_experiment,
            topic="forward_velocity_profiles",
        )
        _save_multi_cell_figure(
            hooks["make_hold_drift_figure"](cell_kms),
            {
                "figure_kind": "hold_drift_profile_pre_go_position_go_aligned",
                "experiment": figure_experiment,
                "inputs": input_artifacts,
                "transform": [
                    shared_eval,
                    {"name": "forward_position_projection_onto_reach_axis", "kwargs": {}},
                    {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
                    {"name": "trim_to_full_support", "kwargs": {"min_coverage": 1.0}},
                    {"name": "replicate_nanmean_over_trials", "kwargs": {}},
                    {"name": "clip_to_pre_go_window", "kwargs": {}},
                ],
                "plot_kwargs": profile_plot,
                "fix_note": (
                    "Bug: 06f7faf — go-cue alignment fix + shared y-axes across cells"
                    + (" (per-replicate variant retains full aligned window)." if lit else ".")
                ),
            },
            experiment=route_experiment,
            topic="hold_drift_profiles",
        )
    if rmse_ratios and cell_stats:
        metric_description = (
            "Primary: velocity-RMSE ratio = within-cell mean pairwise RMSE / "
            "nearest-across-cell mean pairwise RMSE on forward-velocity profiles. "
            "Secondary: same on forward-position profiles. Auxiliary: CV = "
            "SD(peak_vel) / mean(peak_vel) across replicates. Target threshold 0.5; "
            + (
                "prior best (baseline GRU/jerk, 2bc95fd): 0.758."
                if lit
                else "prior best (baseline GRU/jerk matrix): 0.758."
            )
        )
        _save_multi_cell_figure(
            hooks["make_rmse_ratio_figure"](rmse_ratios, cell_stats),
            {
                "figure_kind": "rmse_ratio_comparison_bar",
                "experiment": figure_experiment,
                "metric_description": metric_description,
                "inputs": input_artifacts,
                "transform": [
                    shared_eval,
                    {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
                    {"name": "replicate_nanmean_over_trials", "kwargs": {}},
                    {"name": "pairwise_profile_rmse_ratio_velocity", "kwargs": {}},
                    {"name": "pairwise_profile_rmse_ratio_position", "kwargs": {}},
                ],
                "plot_kwargs": shared_plot,
                "fix_note": "Bug: 06f7faf — go-cue alignment fix.",
                "rmse_ratios": {
                    label: {
                        key: row[key]
                        for key in (
                            "vel_within_rmse",
                            "vel_nearest_across_rmse",
                            "vel_rmse_ratio",
                            "pos_within_rmse",
                            "pos_nearest_across_rmse",
                            "pos_rmse_ratio",
                        )
                    }
                    for label, row in rmse_ratios.items()
                },
            },
            experiment=route_experiment,
            topic="rmse_ratio_comparison",
        )
    _write_multi_cell_report(
        profile=profile,
        experiment=experiment,
        args=args,
        labels=labels,
        display_names=display_names,
        cell_stats=cell_stats,
        rmse_ratios=rmse_ratios,
        notes_dir=notes_dir,
    )


def _save_multi_cell_figure(
    figure: Any,
    spec: Mapping[str, Any],
    *,
    experiment: str,
    topic: str,
) -> None:
    output = save_figure(
        fig=figure,
        spec=dict(spec),
        package="rlrmp",
        experiment=experiment,
        topic=topic,
        extra_packages=["rlrmp"],
    )
    print(f"  Spec: {output['spec_path']}")
    print(f"  Render: {output['render_path']}")


def _write_multi_cell_report(
    *,
    profile: str,
    experiment: str,
    args: argparse.Namespace,
    labels: tuple[str, ...],
    display_names: Mapping[str, str],
    cell_stats: Mapping[str, Mapping[str, Any]],
    rmse_ratios: Mapping[str, Mapping[str, Any]],
    notes_dir: Path,
) -> None:
    """Write the profile-specific table, decision note, and JSON sidecar."""

    lit = profile == "lit_replication"
    prior_best = 0.758
    threshold = 0.5
    winners = [
        label
        for label in labels
        if label in rmse_ratios
        and not np.isnan(rmse_ratios[label]["vel_rmse_ratio"])
        and rmse_ratios[label]["vel_rmse_ratio"] < threshold
    ]
    heading = (
        "=== VARIANCE ANALYSIS SUMMARY (f47abb1 lit-replication) ==="
        if lit
        else "=== VARIANCE ANALYSIS SUMMARY ==="
    )
    display_width = 28 if lit else 42
    print(f"\n{heading}\n")
    header = (
        f"{'Cell':{display_width}s} {'Vel-RMSE-ratio':>15} {'Pos-RMSE-ratio':>15} "
        f"{'CV (peak vel)':>14} {'Mean PV (m/s)':>14} {'Hold drift (mm)':>16} {'TTP':>6}"
    )
    print(header)
    print("-" * len(header))
    for label in labels:
        if label not in cell_stats:
            print(f"  {display_names[label]:{display_width}s} SKIPPED")
            continue
        stats = cell_stats[label]
        row = rmse_ratios.get(label, {})
        vel_ratio = row.get("vel_rmse_ratio", float("nan"))
        pos_ratio = row.get("pos_rmse_ratio", float("nan"))
        flag = " <-- WINNER" if label in winners else ""
        if not lit and label == "gru__jerk" and not flag:
            flag = " (prior best)"
        print(
            f"  {display_names[label]:{display_width}s} {vel_ratio:>15.3f} "
            f"{pos_ratio:>15.3f} {stats['cv_peak_vel']:>14.3f} "
            f"{stats['mean_peak_velocity']:>14.4f} "
            f"{stats['mean_hold_drift_mm']:>16.3f} "
            f"{stats['mean_time_to_peak_steps']:>6.1f}{flag}"
        )

    lines = _report_preamble(profile, experiment, args.sisu)
    lines.extend(
        [
            "## Results Table",
            "",
            "| Cell | Display Name | Vel-RMSE ratio (PRIMARY) | Pos-RMSE ratio | "
            "CV (peak vel) | Mean PV (m/s) | SD PV (m/s) | Hold Drift (mm) | TTP (steps) |",
            "|------|------|---------|---------|---------|---------|---------|---------|---------|",
        ]
    )
    for label in labels:
        if label not in cell_stats:
            lines.append(f"| {label} | {display_names[label]} | SKIPPED | - | - | - | - | - | - |")
            continue
        stats = cell_stats[label]
        row = rmse_ratios.get(label, {})
        vel_ratio = row.get("vel_rmse_ratio", float("nan"))
        pos_ratio = row.get("pos_rmse_ratio", float("nan"))
        vel_text = f"{vel_ratio:.3f}{' *' if label in winners else ''}" if not np.isnan(vel_ratio) else "n/a"
        pos_text = f"{pos_ratio:.3f}" if not np.isnan(pos_ratio) else "n/a"
        lines.append(
            f"| {label} | {display_names[label]} | {vel_text} | {pos_text} | "
            f"{stats['cv_peak_vel']:.3f} | {stats['mean_peak_velocity']:.4f} | "
            f"{stats['sd_peak_velocity']:.4f} | {stats['mean_hold_drift_mm']:.3f} ± "
            f"{stats['sd_hold_drift_mm']:.3f} | {stats['mean_time_to_peak_steps']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* = beats primary threshold (vel-RMSE ratio < 0.50).",
            "",
            "## RMSE Detail (within vs across)",
            "",
            "| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) | "
            "Pos within-RMSE (m) | Pos nearest-across-RMSE (m) |",
            "|------|---------|---------|---------|---------|",
        ]
    )
    for label in labels:
        if label not in rmse_ratios:
            continue
        row = rmse_ratios[label]
        lines.append(
            f"| {label} | {row['vel_within_rmse']:.4f} | "
            f"{row['vel_nearest_across_rmse']:.4f} | {row['pos_within_rmse']:.4f} | "
            f"{row['pos_nearest_across_rmse']:.4f} |"
        )
    lines.extend(["", "## Decision", "", f"**Primary threshold**: vel-RMSE ratio < {threshold}"])
    if winners:
        lines.extend(["", f"**{len(winners)} cell(s) met the threshold:**", ""])
        for label in winners:
            stats = cell_stats[label]
            row = rmse_ratios[label]
            lines.append(
                f"- **{display_names[label]}** (`{label}`): vel-RMSE-ratio = "
                f"{row['vel_rmse_ratio']:.3f}, CV = {stats['cv_peak_vel']:.3f}, "
                f"mean PV = {stats['mean_peak_velocity']:.4f} m/s"
            )
    else:
        candidates = [label for label in cell_stats if label in rmse_ratios]
        best = min(candidates, key=lambda label: rmse_ratios[label]["vel_rmse_ratio"], default=None)
        lines.extend(["", "**No cell met the threshold.**"])
        if best is not None:
            lines.append(
                f"Best cell: **{display_names[best]}** (`{best}`) with vel-RMSE-ratio = "
                f"{rmse_ratios[best]['vel_rmse_ratio']:.3f}"
            )
    lines.extend(
        [
            "",
            "## Anticipation (Hold Drift)",
            "",
            "Hold drift = max forward displacement (toward target, in mm) before the go cue.",
            "Positive = anticipatory movement.",
            "",
        ]
    )
    for label in labels:
        if label in cell_stats:
            stats = cell_stats[label]
            lines.append(
                f"- {display_names[label]}: {stats['mean_hold_drift_mm']:.3f} ± "
                f"{stats['sd_hold_drift_mm']:.3f} mm"
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/rmse_ratio_comparison/` — Bar chart of RMSE ratios (PRIMARY)",
            "- `figures/peak_velocity_distributions/` — Violin/strip plot (CV annotated, auxiliary)",
            "- `figures/forward_velocity_profiles/` — Forward velocity time series per cell",
            "- `figures/hold_drift_profiles/` — Pre-go forward position (anticipation drift)",
        ]
    )
    notes_path = notes_dir / "variance_analysis.md"
    update_marked_section(notes_path, "variance_analysis", "\n".join(lines) + "\n")
    print(f"\nSaved analysis notes: {notes_path}")
    payload = {
        **({"experiment": experiment} if lit else {}),
        "sisu": args.sisu,
        "primary_metric": "vel_rmse_ratio",
        "prior_best_vel_rmse_ratio": prior_best,
        "winner_threshold": threshold,
        "winners": winners,
        "cells": dict(cell_stats),
        "rmse_ratios": {
            label: {
                key: None if isinstance(value, float) and np.isnan(value) else value
                for key, value in row.items()
            }
            for label, row in rmse_ratios.items()
        },
    }
    stats_path = notes_dir / "variance_analysis_data.json"
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved stats JSON: {stats_path}")
    print("\nDone.")


def _report_preamble(profile: str, experiment: str, sisu: float) -> list[str]:
    if profile == "lit_replication":
        return [
            f"# Variance Analysis — Lit-Replication 6-Cell Matrix ({experiment})",
            "",
            "## Setup",
            "",
            "This matrix tests whether faithful replication of the Chaisanguanthum & Shenoy "
            "2019 loss schedule improves velocity-RMSE ratios and inter-replicate variance.",
            "",
            f"- Experiment hash: `{experiment}`",
            f"- SISU: {sisu}",
            "- Perturbation: 0 (clean reach)",
            "- Validation trials: 8 center-out reach directions",
            "",
            "## Metrics",
            "",
            "**Primary: velocity-RMSE ratio** — within-cell mean pairwise RMSE on the "
            "forward-velocity profile / nearest-neighbor across-cell mean pairwise RMSE.",
            "Prior best (GRU/jerk, 2bc95fd): 0.758. Decision threshold: < 0.50.",
            "",
        ]
    return [
        "# Variance Analysis — 6-Cell Anti-Anticipation Matrix",
        "",
        "## Setup",
        f"- SISU: {sisu}",
        "- Perturbation: 0 (clean reach)",
        "- Validation trials: 8 center-out reach directions",
        "",
        "## Metrics",
        "",
        "**Primary: velocity-RMSE ratio** — within-cell mean pairwise RMSE on the "
        "forward-velocity profile / nearest-neighbor across-cell mean pairwise RMSE.",
        "Prior best (GRU/jerk): 0.758. Decision threshold: < 0.50.",
        "",
    ]
