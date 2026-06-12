"""Evaluate all center-out models for the experiment log update.

Ten models (all in results/2ef67ca/models/):
  1.  centerout_baseline_std_pert0       — std, pert_std=0, no update
  2.  centerout_baseline_std_pert1       — std, pert_std=1, no update
  3.  centerout_std_update_r03_pert0     — std+update r=0.3, pert_std=0
  4.  centerout_std_update_r03_pert1     — std+update r=0.3, pert_std=1
  5.  baseline_std_update_r05_pert0      — std+update r=0.5, pert_std=0
  6.  baseline_apt_update_r03_pert0      — APT+update r=0.3, pert_std=0
  7.  centerout_apt_pert1                — APT, pert_std=1, no update
  8.  centerout_apt_update_r03_pert1     — APT+update r=0.3, pert_std=1
  9.  centerout_std_update_r05_pert1     — std+update r=0.5, pert_std=1
  10. centerout_apt_update_r05_pert1     — APT+update r=0.5, pert_std=1

Table 1: Basic metrics + SISU velocity (pert_scale=0, SISU=0.5, 0, 1)
Table 2: Robustness under perturbation (pert_scale=5, SISU=0.5)
         Reference task for Table 2: centerout_baseline_std_pert1

Usage:
    uv run python scripts/eval_centerout_full.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np


from rlrmp.train.standard import build_hps
from rlrmp.eval import (
    N_REPLICATES,
    compute_kinematics,
    eval_ensemble_on_trials,
    set_sisu,
)
from feedbax._io import load_with_hyperparameters
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

WORKTREE = Path(__file__).parent.parent
RESULTS_BASE = WORKTREE / "results" / "2ef67ca"  # legacy Part 2.5 archive (Bug: f485c26)
MODELS_BASE = RESULTS_BASE / "models"

# (index, method, pert_std, update, ratio, model_dir)
CONDITIONS = [
    (1,  "std", 0, False, None, "centerout_baseline_std_pert0"),
    (2,  "std", 1, False, None, "centerout_baseline_std_pert1"),
    (3,  "std", 0, True,  0.3,  "centerout_std_update_r03_pert0"),
    (4,  "std", 1, True,  0.3,  "centerout_std_update_r03_pert1"),
    (5,  "std", 0, True,  0.5,  "baseline_std_update_r05_pert0"),
    (6,  "APT", 0, True,  0.3,  "baseline_apt_update_r03_pert0"),
    (7,  "APT", 1, False, None, "centerout_apt_pert1"),
    (8,  "APT", 1, True,  0.3,  "centerout_apt_update_r03_pert1"),
    (9,  "std", 1, True,  0.5,  "centerout_std_update_r05_pert1"),
    (10, "APT", 1, True,  0.5,  "centerout_apt_update_r05_pert1"),
]

# Model dir used as reference task for Table 2 perturbation evaluation
REFERENCE_TASK_DIR = "centerout_baseline_std_pert1"
PERT_SCALE_TABLE2 = 5.0
SISU_TABLE2 = 0.5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_condition(model_dir_name: str):
    """Load task and trained model for a condition.

    Returns:
        Tuple of (task, trained_model, config), or None if not found.
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


def eval_at_pert0(task, model, sisu: float, *, key):
    """Evaluate with pert_scale=0 at a given SISU level.

    Returns:
        km: dict with "peak_velocity", "endpoint_error", "max_lateral_deviation"
    """
    val_trials = task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.zeros(pert_shape),
    )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


def eval_at_pert_scale(model, ref_task, sisu: float, pert_scale: float, *, key):
    """Evaluate a model on a reference task's trials at given pert_scale and SISU.

    The model is evaluated on the reference task's validation trials (with
    pert_scale set to the given value), so different models are compared on
    an identical set of perturbations.

    Returns:
        km: dict with "peak_velocity", "endpoint_error", "max_lateral_deviation"
    """
    val_trials = ref_task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    # Set pert_scale uniformly across all trials
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.full(pert_shape, pert_scale),
    )
    states = eval_ensemble_on_trials(ref_task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading all conditions...\n")
    loaded = {}
    for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
        label = f"#{idx:>2} {method} pert={pert_std} upd={update} r={ratio}"
        print(f"  {label} ({model_dir}) ... ", end="", flush=True)
        result = load_condition(model_dir)
        if result is None:
            print("SKIPPED — not found")
            loaded[idx] = None
        else:
            print("OK")
            loaded[idx] = result
    print()

    # Load reference task for Table 2
    print(f"Loading reference task ({REFERENCE_TASK_DIR}) for Table 2 ... ", end="", flush=True)
    ref_result = load_condition(REFERENCE_TASK_DIR)
    if ref_result is None:
        print("NOT FOUND — Table 2 will be skipped")
        ref_task = None
    else:
        ref_task, _, _ = ref_result
        print("OK")
    print()

    # =========================================================================
    # Table 1: Basic metrics + SISU vel (pert_scale=0)
    # =========================================================================
    print("=" * 110)
    print("TABLE 1: Basic metrics + SISU velocity (pert_scale=0)")
    print("=" * 110)

    t1 = []
    for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
        entry = loaded[idx]
        if entry is None:
            t1.append(dict(idx=idx, status="missing"))
            continue
        task, model, config = entry
        key = jr.PRNGKey(10 + idx)

        key, k05 = jr.split(key)
        km05 = eval_at_pert0(task, model, sisu=0.5, key=k05)

        key, k0 = jr.split(key)
        km0 = eval_at_pert0(task, model, sisu=0.0, key=k0)

        key, k1 = jr.split(key)
        km1 = eval_at_pert0(task, model, sisu=1.0, key=k1)

        t1.append(dict(
            idx=idx,
            method=method,
            pert_std=pert_std,
            update=update,
            ratio=ratio,
            status="ok",
            vel_05=float(np.mean(km05["peak_velocity"])),
            ep_err=float(np.mean(km05["endpoint_error"])),
            vel_0=float(np.mean(km0["peak_velocity"])),
            vel_1=float(np.mean(km1["peak_velocity"])),
        ))

    print()
    hdr1 = (
        f"{'#':>2} | {'method':>5} | {'pert_std':>8} | {'loss_upd':>8} | {'ratio':>5} | "
        f"{'vel(S=0.5)':>10} | {'ep_err':>8} | {'vel(S=0)':>8} | {'vel(S=1)':>8} | {'Δvel':>8}"
    )
    print(hdr1)
    print("-" * len(hdr1))
    for r in t1:
        if r["status"] == "missing":
            print(f"{r['idx']:>2} | MISSING")
            continue
        upd_str = "yes" if r["update"] else "no"
        ratio_str = f"{r['ratio']}" if r["ratio"] is not None else "—"
        delta = r["vel_1"] - r["vel_0"]
        print(
            f"{r['idx']:>2} | {r['method']:>5} | {r['pert_std']:>8.0f} | {upd_str:>8} | "
            f"{ratio_str:>5} | {r['vel_05']:>10.4f} | {r['ep_err']:>8.4f} | "
            f"{r['vel_0']:>8.4f} | {r['vel_1']:>8.4f} | {delta:>+8.4f}"
        )
    print()

    # =========================================================================
    # Table 2: Robustness under perturbation (pert_scale=5, SISU=0.5)
    # =========================================================================
    print("=" * 110)
    print(f"TABLE 2: Robustness under perturbation (pert_scale={PERT_SCALE_TABLE2}, SISU={SISU_TABLE2})")
    print(f"         Reference task: {REFERENCE_TASK_DIR}")
    print("=" * 110)

    if ref_task is None:
        print("  SKIPPED — reference task not available\n")
    else:
        t2 = []
        for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
            entry = loaded[idx]
            if entry is None:
                t2.append(dict(idx=idx, status="missing"))
                continue
            _, model, _ = entry
            key = jr.PRNGKey(50 + idx)
            km = eval_at_pert_scale(
                model, ref_task, sisu=SISU_TABLE2, pert_scale=PERT_SCALE_TABLE2, key=key
            )
            t2.append(dict(
                idx=idx,
                method=method,
                pert_std=pert_std,
                update=update,
                ratio=ratio,
                status="ok",
                vel=float(np.mean(km["peak_velocity"])),
                lat_dev=float(np.mean(km["max_lateral_deviation"])),
                ep_err=float(np.mean(km["endpoint_error"])),
            ))

        print()
        hdr2 = (
            f"{'#':>2} | {'method':>5} | {'pert_std':>8} | {'loss_upd':>8} | {'ratio':>5} | "
            f"{'vel(p=5)':>8} | {'lat_dev(p=5)':>12} | {'ep_err(p=5)':>11}"
        )
        print(hdr2)
        print("-" * len(hdr2))
        for r in t2:
            if r["status"] == "missing":
                print(f"{r['idx']:>2} | MISSING")
                continue
            upd_str = "yes" if r["update"] else "no"
            ratio_str = f"{r['ratio']}" if r["ratio"] is not None else "—"
            print(
                f"{r['idx']:>2} | {r['method']:>5} | {r['pert_std']:>8.0f} | {upd_str:>8} | "
                f"{ratio_str:>5} | {r['vel']:>8.4f} | {r['lat_dev']:>12.4f} | {r['ep_err']:>11.4f}"
            )
        print()

    print("Notes:")
    print("  - Table 1: ALL evaluations use pert_scale=0 (perturbation amplitude zeroed).")
    print("  - Table 2: pert_scale=5 using reference task trials (centerout_baseline_std_pert1).")
    print("  - SISU (state-informed signal update): 0=no signal, 1=full signal.")
    print("  - Peak velocity = max speed (L2 norm) after go cue, mean over trials & replicates.")
    print("  - Endpoint error = distance from final position to target, mean over trials & replicates.")
    print("  - Lateral deviation = max off-axis displacement after go cue, mean over trials & replicates.")


if __name__ == "__main__":
    main()
