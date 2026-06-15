# ruff: noqa: E402
"""Evaluate ratio sweep models (r=0.1..0.6) plus baseline for control cost ratio analysis.

Conditions (all std method, pert_std=1, loss_update=yes except baseline):
  - centerout_baseline_std_pert1     — baseline, no update
  - centerout_std_update_r01_pert1   — r=0.1
  - centerout_std_update_r02_pert1   — r=0.2
  - centerout_std_update_r03_pert1   — r=0.3
  - centerout_std_update_r04_pert1   — r=0.4
  - centerout_std_update_r05_pert1   — r=0.5
  - centerout_std_update_r06_pert1   — r=0.6

Table 1: pert_scale=0, SISU in {0, 0.5, 1}
Table 2: pert_scale=5, SISU=0.5 (reference task = centerout_baseline_std_pert1)

Usage:
    uv run python results/2ef67ca/scripts/eval_ratio_sweep.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax_cookbook import load_with_hyperparameters

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.eval import (
    compute_kinematics,
    eval_ensemble_on_trials,
    set_sisu,
)
from rlrmp.train.standard import build_hps
from rlrmp.train.task_model import setup_task_model_pair

RESULTS_BASE = Path(__file__).resolve().parent.parent
WORKTREE = RESULTS_BASE.parent.parent
MODELS_BASE = RESULTS_BASE / "models"

# (index, ratio_label, ratio_value, update, model_dir)
CONDITIONS = [
    (0, "baseline", None, False, "centerout_baseline_std_pert1"),
    (1, "0.1",      0.1,  True,  "centerout_std_update_r01_pert1"),
    (2, "0.2",      0.2,  True,  "centerout_std_update_r02_pert1"),
    (3, "0.3",      0.3,  True,  "centerout_std_update_r03_pert1"),
    (4, "0.4",      0.4,  True,  "centerout_std_update_r04_pert1"),
    (5, "0.5",      0.5,  True,  "centerout_std_update_r05_pert1"),
    (6, "0.6",      0.6,  True,  "centerout_std_update_r06_pert1"),
]

REFERENCE_TASK_DIR = "centerout_baseline_std_pert1"
PERT_SCALE_TABLE2 = 5.0
SISU_TABLE2 = 0.5


def load_condition(model_dir_name: str):
    cond_dir = MODELS_BASE / model_dir_name
    config_path = cond_dir / "config.json"
    model_path = cond_dir / "trained_model.eqx"
    if not config_path.exists() or not model_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    trained_model, _ = load_with_hyperparameters(
        model_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    return pair.task, trained_model, config


def eval_at_pert0(task, model, sisu: float, *, key):
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
    val_trials = ref_task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.full(pert_shape, pert_scale),
    )
    states = eval_ensemble_on_trials(ref_task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


def main():
    print("Loading conditions...\n")
    loaded = {}
    for idx, ratio_label, ratio_val, update, model_dir in CONDITIONS:
        tag = f"r={ratio_label}" if update else "baseline"
        print(f"  {tag} ({model_dir}) ... ", end="", flush=True)
        result = load_condition(model_dir)
        if result is None:
            print("SKIPPED — not found")
            loaded[idx] = None
        else:
            print("OK")
            loaded[idx] = result
    print()

    # Reference task for Table 2 (same as baseline, already loaded)
    ref_task = None
    if loaded[0] is not None:
        ref_task, _, _ = loaded[0]
        print(f"Reference task ({REFERENCE_TASK_DIR}): OK\n")
    else:
        print(f"Reference task ({REFERENCE_TASK_DIR}): NOT FOUND — Table 2 will be skipped\n")

    # =========================================================================
    # Table 1: pert_scale=0, SISU in {0, 0.5, 1}
    # =========================================================================
    print("=" * 100)
    print("TABLE 1: SISU velocity effect (pert_scale=0)")
    print("=" * 100)

    t1 = []
    for idx, ratio_label, ratio_val, update, model_dir in CONDITIONS:
        entry = loaded[idx]
        if entry is None:
            t1.append(dict(idx=idx, ratio_label=ratio_label, status="missing"))
            continue
        task, model, config = entry
        key = jr.PRNGKey(100 + idx)

        key, k05 = jr.split(key)
        km05 = eval_at_pert0(task, model, sisu=0.5, key=k05)

        key, k0 = jr.split(key)
        km0 = eval_at_pert0(task, model, sisu=0.0, key=k0)

        key, k1 = jr.split(key)
        km1 = eval_at_pert0(task, model, sisu=1.0, key=k1)

        vel_05 = float(np.mean(km05["peak_velocity"]))
        ep_err = float(np.mean(km05["endpoint_error"]))
        vel_0  = float(np.mean(km0["peak_velocity"]))
        vel_1  = float(np.mean(km1["peak_velocity"]))
        delta  = vel_1 - vel_0

        t1.append(dict(
            idx=idx,
            ratio_label=ratio_label,
            update=update,
            status="ok",
            vel_05=vel_05,
            ep_err=ep_err,
            vel_0=vel_0,
            vel_1=vel_1,
            delta=delta,
            delta_pct=100.0 * delta / vel_0 if vel_0 > 0 else float("nan"),
        ))

    print()
    hdr1 = (
        f"{'#':>2} | {'ratio':>8} | {'loss_upd':>8} | "
        f"{'vel(S=0.5)':>10} | {'ep_err':>8} | {'vel(S=0)':>8} | "
        f"{'vel(S=1)':>8} | {'Δvel':>8} | {'Δvel%':>7}"
    )
    print(hdr1)
    print("-" * len(hdr1))
    for r in t1:
        if r["status"] == "missing":
            print(f"{r['idx']:>2} | r={r['ratio_label']:>5} | MISSING")
            continue
        upd_str = "yes" if r["update"] else "no"
        print(
            f"{r['idx']:>2} | {r['ratio_label']:>8} | {upd_str:>8} | "
            f"{r['vel_05']:>10.4f} | {r['ep_err']:>8.4f} | {r['vel_0']:>8.4f} | "
            f"{r['vel_1']:>8.4f} | {r['delta']:>+8.4f} | {r['delta_pct']:>6.1f}%"
        )
    print()

    # =========================================================================
    # Table 2: pert_scale=5, SISU=0.5
    # =========================================================================
    print("=" * 100)
    print(f"TABLE 2: Robustness under perturbation (pert_scale={PERT_SCALE_TABLE2}, SISU={SISU_TABLE2})")
    print(f"         Reference task: {REFERENCE_TASK_DIR}")
    print("=" * 100)

    if ref_task is None:
        print("  SKIPPED — reference task not available\n")
    else:
        t2 = []
        for idx, ratio_label, ratio_val, update, model_dir in CONDITIONS:
            entry = loaded[idx]
            if entry is None:
                t2.append(dict(idx=idx, ratio_label=ratio_label, status="missing"))
                continue
            _, model, _ = entry
            key = jr.PRNGKey(200 + idx)
            km = eval_at_pert_scale(
                model, ref_task, sisu=SISU_TABLE2, pert_scale=PERT_SCALE_TABLE2, key=key
            )
            t2.append(dict(
                idx=idx,
                ratio_label=ratio_label,
                update=update,
                status="ok",
                vel=float(np.mean(km["peak_velocity"])),
                lat_dev=float(np.mean(km["max_lateral_deviation"])),
                ep_err=float(np.mean(km["endpoint_error"])),
            ))

        print()
        hdr2 = (
            f"{'#':>2} | {'ratio':>8} | {'loss_upd':>8} | "
            f"{'vel(p=5)':>8} | {'lat_dev(p=5)':>12} | {'ep_err(p=5)':>11}"
        )
        print(hdr2)
        print("-" * len(hdr2))
        for r in t2:
            if r["status"] == "missing":
                print(f"{r['idx']:>2} | r={r['ratio_label']:>5} | MISSING")
                continue
            upd_str = "yes" if r["update"] else "no"
            print(
                f"{r['idx']:>2} | {r['ratio_label']:>8} | {upd_str:>8} | "
                f"{r['vel']:>8.4f} | {r['lat_dev']:>12.4f} | {r['ep_err']:>11.4f}"
            )
        print()

    print("Notes:")
    print("  - ratio = control cost / goal state cost fraction target during training.")
    print("  - Table 1: ALL evaluations use pert_scale=0 (perturbation amplitude zeroed).")
    print("  - Table 2: pert_scale=5 using reference task trials (centerout_baseline_std_pert1).")
    print("  - SISU: 0=no signal, 0.5=half signal, 1=full signal.")
    print("  - Peak velocity = max speed after go cue, mean over trials & replicates.")
    print("  - Δvel = vel(S=1) - vel(S=0); Δvel% relative to vel(S=0).")


if __name__ == "__main__":
    main()
