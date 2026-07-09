# ruff: noqa: E402
"""Evaluate fine ratio sweep models (r=0.1, 0.12, 0.15, 0.18, 0.2) for SISU velocity transition.

All conditions: standard method, pert_std=1, loss_update=yes.
All evaluations at pert_scale=0.

Table columns: # | ratio | vel(S=0.5) | ep_err | vel(S=0) | vel(S=1) | Δvel | Δvel%

Usage:
    uv run python results/2ef67ca/scripts/eval_fine_ratio_sweep.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import jax.random as jr
import numpy as np
from jax_cookbook import load_with_hyperparameters

from rlrmp.eval.pert import eval_at_pert0 as _canonical_eval_at_pert0
from rlrmp.train.standard import build_hps
from rlrmp.train.task_model import setup_task_model_pair

RESULTS_BASE = Path(__file__).resolve().parent.parent
WORKTREE = RESULTS_BASE.parent.parent
MODELS_BASE = RESULTS_BASE / "models"

CONDITIONS = [
    (0, "0.10", "centerout_std_update_r01_pert1"),
    (1, "0.12", "centerout_std_update_r012_pert1"),
    (2, "0.15", "centerout_std_update_r015_pert1"),
    (3, "0.18", "centerout_std_update_r018_pert1"),
    (4, "0.20", "centerout_std_update_r02_pert1"),
]


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
    return _canonical_eval_at_pert0(task, model, sisu, key=key)


def main():
    print("Loading conditions...\n")
    loaded = {}
    for idx, ratio_label, model_dir in CONDITIONS:
        print(f"  r={ratio_label} ({model_dir}) ... ", end="", flush=True)
        result = load_condition(model_dir)
        if result is None:
            print("SKIPPED — not found")
            loaded[idx] = None
        else:
            print("OK")
            loaded[idx] = result
    print()

    print("=" * 80)
    print("Fine ratio sweep: SISU velocity effect (pert_scale=0)")
    print("=" * 80)

    rows = []
    for idx, ratio_label, model_dir in CONDITIONS:
        entry = loaded[idx]
        if entry is None:
            rows.append(dict(idx=idx, ratio_label=ratio_label, status="missing"))
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

        rows.append(dict(
            idx=idx,
            ratio_label=ratio_label,
            status="ok",
            vel_05=vel_05,
            ep_err=ep_err,
            vel_0=vel_0,
            vel_1=vel_1,
            delta=delta,
            delta_pct=100.0 * delta / vel_0 if vel_0 > 0 else float("nan"),
        ))

    print()
    hdr = (
        f"{'#':>2} | {'ratio':>6} | "
        f"{'vel(S=0.5)':>10} | {'ep_err':>8} | {'vel(S=0)':>8} | "
        f"{'vel(S=1)':>8} | {'Δvel':>8} | {'Δvel%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["status"] == "missing":
            print(f"{r['idx']:>2} | r={r['ratio_label']:>5} | MISSING")
            continue
        print(
            f"{r['idx']:>2} | {r['ratio_label']:>6} | "
            f"{r['vel_05']:>10.4f} | {r['ep_err']:>8.4f} | {r['vel_0']:>8.4f} | "
            f"{r['vel_1']:>8.4f} | {r['delta']:>+8.4f} | {r['delta_pct']:>6.1f}%"
        )
    print()
    print("Notes:")
    print("  - All evals at pert_scale=0 (perturbation amplitude zeroed).")
    print("  - SISU: 0=no signal, 0.5=half signal, 1=full signal.")
    print("  - Peak velocity = max speed after go cue, mean over trials & replicates.")
    print("  - Δvel = vel(S=1) - vel(S=0); Δvel% relative to vel(S=0).")


if __name__ == "__main__":
    main()
