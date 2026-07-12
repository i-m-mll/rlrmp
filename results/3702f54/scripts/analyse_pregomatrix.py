# TODO: relocate to results/3702f54/scripts/ — per CLAUDE.md script-placement convention
"""Pre-go motor mask follow-up matrix analysis (3702f54).

Tests whether (a) scaling position weight 10x or (b) the `--nn-output-pre-go` lever
can suppress the residual ~2-3 mm pre-go anticipation observed in f47abb1's leading
no-jerk powerlaw cells (`lit__post_nojerk`, `lit__full_nojerk`).

Bug: 06f7faf — go-cue alignment fix. Forward velocity profiles, hold drift
profiles, and the within-cell vel-RMSE metric are now computed on
per-trial go-cue-aligned profiles using `rlrmp.analysis.math.trial_alignment`.

Loads 10 models:
  - 8 new cells from `_artifacts/3702f54/runs/<cell>/warmup_model.eqx`
  - 2 baseline anchors from `_artifacts/f47abb1/runs/{lit__post_nojerk,lit__full_nojerk}/warmup_model.eqx`

Computes per-(cell x replicate):
  - Within-cell pairwise velocity-RMSE (m/s, absolute) -- primary frame
  - Hold drift mm (RMS forward effector position during hold period)
  - Pre-go forward drift specifically over the [-200ms, 0] pre-go window
  - Peak forward velocity (m/s)
  - Time-to-peak velocity (steps after go)

The five historical figures are now rendered from native manifest-bound FigureSpecs.
Baselines from f47abb1 (`lit__post_nojerk`, `lit__full_nojerk`) are placed at the TOP
of cell ordering / first columns for direct visual comparison.

For cells where n_adversary_batches=0 the canonical final model is `warmup_model.eqx`
(Bug: a517040). Bug: f47abb1.

Usage (from repo root):
    .venv/bin/python scripts/analyse_pregomatrix.py
"""

from __future__ import annotations
from rlrmp.analysis.multi_cell_driver import (
    args_namespace,
    compute_kinematics_per_replicate,
)

import argparse
import json
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.math.trial_alignment import (
    align_trials,
    replicate_mean_curves,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.paths import REPO_ROOT  # Bug: 8404108 — was __file__-relative
from rlrmp.train.minimax_native import build_hps
from rlrmp.train.task_model import setup_task_model_pair

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

# Baselines at TOP for direct visual comparison (per user request).
BASELINE_CELLS = [
    "lit__post_nojerk",
    "lit__full_nojerk",
]

NEW_CELLS = [
    "post_go_pl__pos10",
    "full_trial_pl__pos10",
    "full_trial_pl__prego_1e-3",
    "full_trial_pl__prego_5e-2",
    "full_trial_pl__prego_1",
    "full_trial_pl__pos10_prego_1e-3",
    "full_trial_pl__pos10_prego_5e-2",
    "full_trial_pl__pos10_prego_1",
]

CELL_LABELS = BASELINE_CELLS + NEW_CELLS

CELL_DISPLAY_NAMES = {
    "lit__post_nojerk":                "[baseline] Post-go PL, no jerk (f47abb1)",
    "lit__full_nojerk":                "[baseline] Full-trial PL, no jerk (f47abb1)",
    "post_go_pl__pos10":               "Post-go PL, pos10",
    "full_trial_pl__pos10":            "Full-trial PL, pos10",
    "full_trial_pl__prego_1e-3":       "Full-trial PL, prego=1e-3",
    "full_trial_pl__prego_5e-2":       "Full-trial PL, prego=5e-2",
    "full_trial_pl__prego_1":          "Full-trial PL, prego=1.0",
    "full_trial_pl__pos10_prego_1e-3": "Full-trial PL, pos10, prego=1e-3",
    "full_trial_pl__pos10_prego_5e-2": "Full-trial PL, pos10, prego=5e-2",
    "full_trial_pl__pos10_prego_1":    "Full-trial PL, pos10, prego=1.0",
}

# Per-cell CLI args that differ from shared defaults. Mirrors the run.json specs.
CELL_EXTRA_ARGS: dict[str, dict] = {
    # f47abb1 baselines
    "lit__post_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "flat",
        "nn_output_pre_go": 0.0,
    },
    "lit__full_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 0.0,
    },
    # New 3702f54 cells
    "post_go_pl__pos10": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "flat",
        "nn_output_pre_go": 0.0,
    },
    "full_trial_pl__pos10": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 0.0,
    },
    "full_trial_pl__prego_1e-3": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1e-3,
    },
    "full_trial_pl__prego_5e-2": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 5e-2,
    },
    "full_trial_pl__prego_1": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1.0,
    },
    "full_trial_pl__pos10_prego_1e-3": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1e-3,
    },
    "full_trial_pl__pos10_prego_5e-2": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 5e-2,
    },
    "full_trial_pl__pos10_prego_1": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1.0,
    },
}

# Cells live in different artifact experiment dirs.
CELL_ARTIFACT_EXPERIMENT = {
    "lit__post_nojerk": "f47abb1",
    "lit__full_nojerk": "f47abb1",
    **{c: "3702f54" for c in NEW_CELLS},
}

# Color palette: baselines in grey shades, then qualitative palette for the matrix.
CELL_COLORS = {
    "lit__post_nojerk":                "#7f7f7f",  # grey
    "lit__full_nojerk":                "#4d4d4d",  # dark grey
    "post_go_pl__pos10":               "#1f77b4",  # blue
    "full_trial_pl__pos10":            "#17becf",  # cyan
    "full_trial_pl__prego_1e-3":       "#ffbb78",  # light orange
    "full_trial_pl__prego_5e-2":       "#ff7f0e",  # orange
    "full_trial_pl__prego_1":          "#d62728",  # red
    "full_trial_pl__pos10_prego_1e-3": "#aec7e8",  # light blue
    "full_trial_pl__pos10_prego_5e-2": "#9467bd",  # purple
    "full_trial_pl__pos10_prego_1":    "#8c564b",  # brown
}

N_REPLICATES = 5
N_WARMUP_BATCHES = 12000
EXPERIMENT = "3702f54"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _make_args_namespace(label: str) -> argparse.Namespace:
    """Build argparse.Namespace with the correct per-cell CLI flags."""
    return args_namespace(
        profile="lit_replication",
        n_warmup_batches=N_WARMUP_BATCHES,
        n_replicates=N_REPLICATES,
        overrides=CELL_EXTRA_ARGS[label],
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cell_model(label: str, artifact_base: Path):
    """Load the warmup model for a cell. `warmup_model.eqx` is the canonical final
    model since n_adversary_batches=0 across this matrix (Bug: a517040).
    """
    experiment = CELL_ARTIFACT_EXPERIMENT[label]
    cell_dir = artifact_base / experiment / "runs" / label
    eqx_path = cell_dir / "warmup_model.eqx"
    if not eqx_path.exists():
        raise FileNotFoundError(f"warmup_model.eqx not found: {eqx_path}")

    args = _make_args_namespace(label)
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(42))
    task = pair.task

    model, _ = load_with_hyperparameters(
        eqx_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    n_reps = _count_replicates(model)
    return model, task, n_reps, eqx_path


def _count_replicates(model) -> int:
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim >= 3:
            return int(leaf.shape[0])
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim == 2:
            return 1
    return 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_zero_pert_trials(task, *, sisu: float = 0.5):
    val = task.validation_trials
    n_trials = val.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val,
        jnp.zeros((n_trials,)),
    )
    trial_specs = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        trial_specs,
        jnp.full((n_trials,), sisu),
    )
    return trial_specs


def eval_ensemble(task, model, trial_specs, *, key: jax.Array, n_replicates: int):
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    def _is_rep(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    arrays, other = eqx.partition(model, _is_rep)

    def _eval_one(arr, oth, rep_key):
        m = eqx.combine(arr, oth)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(m, trial_specs, keys)

    rep_keys = jr.split(key, n_replicates)
    states = eqx.filter_vmap(_eval_one, in_axes=(0, None, 0))(arrays, other, rep_keys)
    return states  # (n_rep, n_trials, n_steps, ...)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

# Pre-go window length in steps (200 ms at dt=10 ms).
PRE_GO_WINDOW_STEPS = 20




# ---------------------------------------------------------------------------
# Per-cell aggregate statistics
# ---------------------------------------------------------------------------

def _within_cell_mean_pairwise_rmse(profiles: np.ndarray) -> float:
    """Mean pairwise RMSE between all distinct pairs of replicates."""
    n_rep = profiles.shape[0]
    rmse_vals = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            diff = profiles[i] - profiles[j]
            rmse_vals.append(float(np.sqrt(np.mean(diff ** 2))))
    return float(np.mean(rmse_vals)) if rmse_vals else float("nan")


def compute_cell_stats(km: dict[str, np.ndarray]) -> dict:
    """Aggregate kinematics to per-replicate scalars + per-cell stats."""
    peak_vel_per_rep = km["peak_forward_velocity"].mean(axis=-1)
    hold_drift_per_rep = km["hold_drift_mm"].mean(axis=-1)
    pre_go_rms_per_rep = km["pre_go_rms_mm"].mean(axis=-1)
    pre_go_mean_per_rep = km["pre_go_mean_mm"].mean(axis=-1)
    ttp_per_rep = km["time_to_peak_after_go"].mean(axis=-1)

    # Within-cell pairwise velocity-RMSE (absolute m/s, primary metric per issue request).
    # Bug: 06f7faf — align per-trial profiles to each trial's go cue BEFORE
    # averaging across trials, and trim to the full-support window so the
    # NaN-edge columns do not contaminate the RMSE.
    aligned_vel, _center = align_trials(km["forward_vel_profile"], km["go_idx"])
    vel_profiles, _sl = replicate_mean_curves(aligned_vel)  # (n_rep, n_kept_steps)
    within_rmse_vel = _within_cell_mean_pairwise_rmse(vel_profiles)

    mean_pv = float(peak_vel_per_rep.mean())
    sd_pv = float(peak_vel_per_rep.std(ddof=1)) if len(peak_vel_per_rep) > 1 else 0.0

    return {
        "peak_vel_per_rep": peak_vel_per_rep.tolist(),
        "hold_drift_per_rep": hold_drift_per_rep.tolist(),
        "pre_go_rms_per_rep": pre_go_rms_per_rep.tolist(),
        "pre_go_mean_per_rep": pre_go_mean_per_rep.tolist(),
        "time_to_peak_per_rep": ttp_per_rep.tolist(),
        "mean_peak_velocity": mean_pv,
        "sd_peak_velocity": sd_pv,
        "mean_hold_drift_mm": float(hold_drift_per_rep.mean()),
        "sd_hold_drift_mm": float(hold_drift_per_rep.std(ddof=1)) if len(hold_drift_per_rep) > 1 else 0.0,
        "mean_pre_go_rms_mm": float(pre_go_rms_per_rep.mean()),
        "sd_pre_go_rms_mm": float(pre_go_rms_per_rep.std(ddof=1)) if len(pre_go_rms_per_rep) > 1 else 0.0,
        "mean_pre_go_drift_mm": float(pre_go_mean_per_rep.mean()),
        "sd_pre_go_drift_mm": float(pre_go_mean_per_rep.std(ddof=1)) if len(pre_go_mean_per_rep) > 1 else 0.0,
        "mean_time_to_peak_steps": float(ttp_per_rep.mean()),
        "sd_time_to_peak_steps": float(ttp_per_rep.std(ddof=1)) if len(ttp_per_rep) > 1 else 0.0,
        "within_cell_vel_rmse": within_rmse_vel,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=None,
        help="Base directory above experiment dirs (default: <repo>/_artifacts)",
    )
    parser.add_argument(
        "--sisu",
        type=float,
        default=0.5,
        help="SISU level for evaluation (default: 0.5)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    artifact_base = args.artifact_base or (REPO_ROOT / "_artifacts")
    results_base = REPO_ROOT / "results" / EXPERIMENT

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")

    notes_dir = results_base / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    cell_stats: dict[str, dict] = {}

    for label in CELL_LABELS:
        print(f"\n[{label}] Loading model ...", flush=True)
        try:
            model, task, n_reps, _eqx_path = load_cell_model(label, artifact_base)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        except Exception as e:
            import traceback
            print(f"  FAILED loading: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        print(f"  Loaded. n_replicates={n_reps}")
        try:
            trial_specs = build_zero_pert_trials(task, sisu=args.sisu)
            n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
            print(f"  Evaluating {n_reps} replicates on {n_trials} trials ...", flush=True)
            states = eval_ensemble(
                task, model, trial_specs,
                key=jr.PRNGKey(args.eval_seed), n_replicates=n_reps,
            )
            print("  Eval OK. Computing kinematics ...", flush=True)
            km = compute_kinematics_per_replicate(states, trial_specs)
            stats = compute_cell_stats(km)
            cell_stats[label] = stats
        except Exception as e:
            import traceback
            print(f"  FAILED eval/kinematics: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        print(
            f"  peak_vel: {stats['mean_peak_velocity']:.4f} +/- {stats['sd_peak_velocity']:.4f} m/s  "
            f"vel-RMSE: {stats['within_cell_vel_rmse']:.4f} m/s  "
            f"hold_drift: {stats['mean_hold_drift_mm']:.2f} mm  "
            f"pre_go_RMS: {stats['mean_pre_go_rms_mm']:.2f} mm  "
            f"TTP: {stats['mean_time_to_peak_steps']:.1f} steps"
        )

    if not cell_stats:
        print("\nNo cells loaded -- aborting.")
        return

    # Figure rendering is declarative; this legacy analysis only computes the summary data.

    # -----------------------------------------------------------------------
    # Summary table + JSON
    # -----------------------------------------------------------------------
    print("\n=== SUMMARY (3702f54 pre-go motor mask matrix) ===\n")
    header = (
        f"{'Cell':36s} {'Vel-RMSE':>10} {'HoldDrift':>10} {'PreGoRMS':>10} "
        f"{'PeakVel':>10} {'TTP':>6}"
    )
    print(header)
    print("-" * len(header))
    for label in CELL_LABELS:
        if label not in cell_stats:
            print(f"  {CELL_DISPLAY_NAMES[label]:36s} SKIPPED")
            continue
        s = cell_stats[label]
        print(
            f"  {CELL_DISPLAY_NAMES[label][:34]:34s}  "
            f"{s['within_cell_vel_rmse']:>9.4f}  "
            f"{s['mean_hold_drift_mm']:>9.2f}  "
            f"{s['mean_pre_go_rms_mm']:>9.2f}  "
            f"{s['mean_peak_velocity']:>9.4f}  "
            f"{s['mean_time_to_peak_steps']:>5.1f}"
        )

    # Save per-cell stats as JSON
    stats_json_path = notes_dir / "analysis_data.json"
    json_data = {
        "experiment": EXPERIMENT,
        "sisu": args.sisu,
        "pre_go_window_steps": PRE_GO_WINDOW_STEPS,
        "baselines": BASELINE_CELLS,
        "new_cells": NEW_CELLS,
        "metric_descriptions": {
            "within_cell_vel_rmse": "Mean pairwise forward-velocity RMSE across replicate pairs within the cell (m/s, absolute).",
            "mean_hold_drift_mm": "Max forward effector displacement during the whole hold period (mm), averaged over replicates.",
            "mean_pre_go_rms_mm": "Root-mean-square forward effector position over the last 200 ms before go (mm).",
            "mean_pre_go_drift_mm": "Mean forward effector position over the last 200 ms before go (mm; signed).",
            "mean_peak_velocity": "Peak forward velocity after go cue (m/s).",
            "mean_time_to_peak_steps": "Steps from go cue to peak velocity (dt=10 ms).",
        },
        "cells": cell_stats,
    }
    with open(stats_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved stats JSON: {stats_json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
