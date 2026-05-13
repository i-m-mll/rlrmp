"""Evaluate pert_std=0 ratio sweep models (r=0.05, 0.1, 0.15) plus baselines.

All evaluations at pert_scale=0.
Compares with known pert_std=1 results at r=0.1 and r=0.15.

Table columns: ratio | pert_std | vel(S=0) | vel(S=1) | Δvel% | ep_err

Usage:
    uv run python scripts/eval_pert0_ratio_sweep.py
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


from rlrmp.train.standard import build_hps
from rlrmp.eval import (
    compute_kinematics,
    eval_ensemble_on_trials,
    set_sisu,
)
from feedbax._io import load_with_hyperparameters
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

WORKTREE = Path(__file__).parent.parent
RESULTS_BASE = WORKTREE / "results" / "part2_5"
MODELS_BASE = RESULTS_BASE / "models"

# (ratio_label, pert_std_label, model_dir_name)
CONDITIONS = [
    ("baseline", "0", "centerout_baseline_std_pert0"),
    ("0.05",     "0", "centerout_std_update_r005_pert0"),
    ("0.10",     "0", "centerout_std_update_r01_pert0"),
    ("0.15",     "0", "centerout_std_update_r015_pert0"),
    ("0.30",     "0", "centerout_std_update_r03_pert0"),
]

# Known pert_std=1 results from earlier fine ratio sweep eval.
# vel(S=0.5) values; vel(S=0) and vel(S=1) not available, so we use S=0.5
# as best proxy for the ratio comparison.
KNOWN_PERT1 = {
    "0.10": {"vel_s05": 1.985},
    "0.15": {"vel_s05": 1.974},
}


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
        cond_dir / "trained_model.eqx",
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


def main():
    print("Loading conditions...\n")
    loaded = {}
    for ratio_label, pert_std_label, model_dir in CONDITIONS:
        key_str = f"{ratio_label}|{pert_std_label}"
        print(f"  r={ratio_label} pert_std={pert_std_label} ({model_dir}) ... ", end="", flush=True)
        result = load_condition(model_dir)
        if result is None:
            print("SKIPPED — not found")
            loaded[key_str] = None
        else:
            print("OK")
            loaded[key_str] = result
    print()

    print("=" * 90)
    print("pert_std=0 ratio sweep: SISU velocity effect (pert_scale=0)")
    print("=" * 90)

    rows = []
    for idx, (ratio_label, pert_std_label, model_dir) in enumerate(CONDITIONS):
        key_str = f"{ratio_label}|{pert_std_label}"
        entry = loaded[key_str]
        if entry is None:
            rows.append(dict(ratio_label=ratio_label, pert_std_label=pert_std_label, status="missing"))
            continue
        task, model, config = entry
        key = jr.PRNGKey(200 + idx)

        key, k0 = jr.split(key)
        km0 = eval_at_pert0(task, model, sisu=0.0, key=k0)

        key, k1 = jr.split(key)
        km1 = eval_at_pert0(task, model, sisu=1.0, key=k1)

        key, k05 = jr.split(key)
        km05 = eval_at_pert0(task, model, sisu=0.5, key=k05)

        vel_0  = float(np.mean(km0["peak_velocity"]))
        vel_1  = float(np.mean(km1["peak_velocity"]))
        vel_05 = float(np.mean(km05["peak_velocity"]))
        ep_err = float(np.mean(km05["endpoint_error"]))
        delta  = vel_1 - vel_0
        delta_pct = 100.0 * delta / vel_0 if vel_0 > 0 else float("nan")

        rows.append(dict(
            ratio_label=ratio_label,
            pert_std_label=pert_std_label,
            status="ok",
            vel_0=vel_0,
            vel_1=vel_1,
            vel_05=vel_05,
            ep_err=ep_err,
            delta=delta,
            delta_pct=delta_pct,
        ))

    print()
    hdr = (
        f"{'ratio':>8} | {'pert_std':>8} | "
        f"{'vel(S=0)':>8} | {'vel(S=1)':>8} | {'Δvel%':>7} | {'ep_err':>8}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in rows:
        if r["status"] == "missing":
            print(f"{r['ratio_label']:>8} | {r['pert_std_label']:>8} | MISSING")
            continue
        print(
            f"{r['ratio_label']:>8} | {r['pert_std_label']:>8} | "
            f"{r['vel_0']:>8.4f} | {r['vel_1']:>8.4f} | {r['delta_pct']:>6.1f}% | "
            f"{r['ep_err']:>8.4f}"
        )

    print()
    print("=" * 90)
    print("Comparison: vel(pert_std=1) / vel(pert_std=0) at S=0.5")
    print("  (pert_std=1 values from earlier fine ratio sweep eval)")
    print("=" * 90)
    print()
    cmp_hdr = f"{'ratio':>8} | {'vel(S=0.5, p=0)':>15} | {'vel(S=0.5, p=1)':>15} | {'ratio p1/p0':>12}"
    print(cmp_hdr)
    print("-" * len(cmp_hdr))
    for r in rows:
        if r["status"] == "missing":
            continue
        ratio_label = r["ratio_label"]
        if ratio_label not in KNOWN_PERT1:
            continue
        vel_p0 = r["vel_05"]
        vel_p1 = KNOWN_PERT1[ratio_label]["vel_s05"]
        ratio_val = vel_p1 / vel_p0 if vel_p0 > 0 else float("nan")
        print(
            f"{ratio_label:>8} | {vel_p0:>15.4f} | {vel_p1:>15.4f} | {ratio_val:>12.4f}"
        )

    print()
    print("Notes:")
    print("  - All evals at pert_scale=0 (perturbation amplitude zeroed).")
    print("  - SISU: 0=no signal, 0.5=half signal, 1=full signal.")
    print("  - Peak velocity = max speed after go cue, mean over trials & replicates.")
    print("  - Δvel% = (vel(S=1) - vel(S=0)) / vel(S=0) * 100.")
    print("  - pert_std=1 vel(S=0.5) values from earlier fine ratio sweep eval.")


if __name__ == "__main__":
    main()
