"""Robustness comparison table across all trained conditions at SISU=0.5.

Evaluates each condition at pert_scale=0 and pert_scale=5 and reports:
  - Peak forward velocity (parallel to target direction)
  - Max lateral deviation (orthogonal to target direction)
  - Endpoint error

The perturbed evaluation uses the reference task from running_cost_standard.
pert_scale=5 was chosen so that the baseline shows clearly visible lateral deviation.

Usage:
    uv run python scripts/eval_robustness_table.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax import load_with_hyperparameters

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.eval import (
    N_REPLICATES,
    compute_kinematics,
    eval_ensemble_on_trials,
    set_sisu,
)
from rlrmp.train.standard import build_hps
from rlrmp.train.task_model import setup_task_model_pair

WORKTREE = Path(__file__).parent.parent
RESULTS_BASE = WORKTREE / "results" / "part2_5"
MODELS_BASE = RESULTS_BASE / "models"

SISU = 0.5
PERT_SCALE = 5.0  # chosen so baseline lateral deviation is clearly visible

# (display_name, model_dir, group, pert_std, loss_upd, method)
# group ordering: baseline, standard, std_update, apt_no_update, apt_update
ALL_CONDITIONS = [
    # --- Baselines ---
    ("baseline_no_pert",       "baseline_no_pert",       "baseline",       0.0,  False, "std"),
    ("running_cost_standard",  "running_cost_standard",  "baseline",       1.0,  False, "std"),
    # --- APT, no loss_update ---
    ("apt_lr001",              "apt_lr001",              "apt_no_update",  1.0,  False, "apt"),
    # --- Standard + loss_update ---
    ("ratio03_pert1_v4",       "ratio03_pert1_v4",       "std_update",     1.0,  True,  "std"),
    ("ratio05_pert1_v4",       "ratio05_pert1_v4",       "std_update",     1.0,  True,  "std"),
    ("ratio03_pert10_v4",      "ratio03_pert10_v4",      "std_update",     10.0, True,  "std"),
    ("ratio05_pert10_v4",      "ratio05_pert10_v4",      "std_update",     10.0, True,  "std"),
    # --- APT + loss_update ---
    ("apt_ratio03_pert1",      "apt_ratio03_pert1",      "apt_update",     1.0,  True,  "apt"),
    ("apt_ratio05_pert1",      "apt_ratio05_pert1",      "apt_update",     1.0,  True,  "apt"),
    ("apt_ratio03_pert10",     "apt_ratio03_pert10",     "apt_update",     10.0, True,  "apt"),
    ("apt_ratio05_pert10",     "apt_ratio05_pert10",     "apt_update",     10.0, True,  "apt"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_condition(model_dir_name: str):
    """Load task and trained model for a single condition.

    Returns:
        Tuple of (task, trained_model, config) or None if not found.
    """
    cond_dir = MODELS_BASE / model_dir_name
    config_path = cond_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    trained_model, _ = load_with_hyperparameters(
        cond_dir / "trained_model.eqx",
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    return pair.task, trained_model, config


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_at_sisu_pert(task, model, sisu: float, pert_scale: float, *, key, ref_task=None):
    """Evaluate model at fixed SISU and perturbation scale.

    Args:
        task: The model's own task (used for model structure).
        model: Trained model.
        sisu: SISU level [0, 1].
        pert_scale: Perturbation scale factor; 0 = no perturbation.
        key: JAX PRNGKey.
        ref_task: Reference task for trial specs when pert_scale > 0.
            Falls back to task if None.

    Returns:
        km: dict with peak_velocity, max_lateral_deviation, endpoint_error.
    """
    source_task = ref_task if (ref_task is not None and pert_scale > 0) else task
    val_trials = source_task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)

    # Set perturbation scale (zero it out when pert_scale=0)
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.full(trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape, pert_scale),
    )

    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Loading reference task (running_cost_standard)...")
    ref_result = load_condition("running_cost_standard")
    if ref_result is None:
        raise RuntimeError("Reference task 'running_cost_standard' not found.")
    ref_task, _, _ = ref_result
    print("  OK\n")

    print("Loading and evaluating all conditions...\n")
    rows = []

    for display_name, model_dir, group, pert_std, loss_upd, method in ALL_CONDITIONS:
        print(f"  [{group}] {display_name} ...")
        result = load_condition(model_dir)
        if result is None:
            print(f"    SKIPPED — not found")
            rows.append({
                "name": display_name, "method": method, "pert_std": pert_std,
                "loss_upd": loss_upd, "group": group, "status": "missing",
            })
            continue

        task, model, config = result
        key = jr.PRNGKey(7)

        # --- Unperturbed eval (pert_scale=0) ---
        key, k0 = jr.split(key)
        km0 = eval_at_sisu_pert(task, model, sisu=SISU, pert_scale=0.0, key=k0)
        vel_p0 = float(np.mean(km0["peak_velocity"]))
        lat_p0 = float(np.mean(km0["max_lateral_deviation"]))
        ep_p0 = float(np.mean(km0["endpoint_error"]))

        # --- Perturbed eval (pert_scale=PERT_SCALE, ref task trials) ---
        key, k1 = jr.split(key)
        km1 = eval_at_sisu_pert(
            task, model, sisu=SISU, pert_scale=PERT_SCALE, key=k1, ref_task=ref_task
        )
        vel_pX = float(np.mean(km1["peak_velocity"]))
        lat_pX = float(np.mean(km1["max_lateral_deviation"]))
        ep_pX = float(np.mean(km1["endpoint_error"]))

        print(
            f"    pert=0: vel={vel_p0:.3f}  lat={lat_p0:.4f}  ep_err={ep_p0:.4f} | "
            f"pert={PERT_SCALE}: vel={vel_pX:.3f}  lat={lat_pX:.4f}  ep_err={ep_pX:.4f}"
        )

        rows.append({
            "name": display_name, "method": method, "pert_std": pert_std,
            "loss_upd": loss_upd, "group": group, "status": "ok",
            "vel_p0": vel_p0, "lat_p0": lat_p0, "ep_p0": ep_p0,
            "vel_pX": vel_pX, "lat_pX": lat_pX, "ep_pX": ep_pX,
        })

    # ---------------------------------------------------------------------------
    # Markdown table
    # ---------------------------------------------------------------------------
    X = int(PERT_SCALE) if PERT_SCALE == int(PERT_SCALE) else PERT_SCALE
    h_vel0 = "vel(p=0)"
    h_velX = f"vel(p={X})"
    h_lat0 = "lat_dev(p=0)"
    h_latX = f"lat_dev(p={X})"
    h_ep0  = "ep_err(p=0)"
    h_epX  = f"ep_err(p={X})"

    print()
    print(f"## Robustness comparison — SISU={SISU}, pert_scale={X} (ref task: running_cost_standard)")
    print()
    print(
        f"| {'Condition':<26} | {'Method':<3} | {'pert_std':>8} | {'loss_upd':>8} | "
        f"{h_vel0:>9} | {h_velX:>9} | {h_lat0:>12} | {h_latX:>12} | "
        f"{h_ep0:>11} | {h_epX:>11} |"
    )
    print(
        f"|{'-'*28}|{'-'*5}|{'-'*10}|{'-'*10}|"
        f"{'-'*11}|{'-'*11}|{'-'*14}|{'-'*14}|"
        f"{'-'*13}|{'-'*13}|"
    )

    group_order = ["baseline", "std_update", "apt_no_update", "apt_update"]
    rows_sorted = sorted(
        rows,
        key=lambda r: (group_order.index(r["group"]) if r["group"] in group_order else 99,
                       r["pert_std"])
    )

    for r in rows_sorted:
        upd_str = "yes" if r["loss_upd"] else "no"
        if r["status"] == "missing":
            print(
                f"| {r['name']:<26} | {r['method']:<3} | {r['pert_std']:>8.1f} | "
                f"{'—':>8} | {'MISSING':>9} | {'—':>9} | {'—':>12} | {'—':>12} | "
                f"{'—':>11} | {'—':>11} |"
            )
        else:
            print(
                f"| {r['name']:<26} | {r['method']:<3} | {r['pert_std']:>8.1f} | "
                f"{upd_str:>8} | {r['vel_p0']:>9.3f} | {r['vel_pX']:>9.3f} | "
                f"{r['lat_p0']:>12.4f} | {r['lat_pX']:>12.4f} | "
                f"{r['ep_p0']:>11.4f} | {r['ep_pX']:>11.4f} |"
            )

    print()
    print(f"**Notes:**")
    print(f"- SISU={SISU} for all evaluations.")
    print(f"- `vel` = peak speed (L2 norm of velocity) after go cue, mean over replicates×trials.")
    print(f"- `lat_dev` = max lateral deviation from straight-line path, mean over replicates×trials.")
    print(f"- `ep_err` = endpoint position error (L2 to goal), mean over replicates×trials.")
    print(f"- `p=0`: perturbation scale set to 0 (no gusts) on each model's own task trials.")
    print(f"- `p={X}`: perturbation scale={X}, using running_cost_standard task's validation trials.")
    print(f"- `loss_upd`: whether adaptive nn_output weight update (enable_loss_update) was used.")


if __name__ == "__main__":
    main()
