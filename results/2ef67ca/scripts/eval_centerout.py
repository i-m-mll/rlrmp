# ruff: noqa: E402
"""Evaluate center-out translation-invariant models.

Key question: does the pert_std=0 baseline now converge with center-out reaches?

Six models:
  1. centerout_baseline_std_pert0  — std, pert_std=0, no update
  2. centerout_baseline_std_pert1  — std, pert_std=1, no update
  3. centerout_std_update_r03_pert0 — std+update r=0.3, pert_std=0
  4. centerout_std_update_r03_pert1 — std+update r=0.3, pert_std=1
  5. baseline_std_update_r05_pert0  — std+update r=0.5, pert_std=0
  6. baseline_apt_update_r03_pert0  — APT+update r=0.3, pert_std=0

Three analyses (all at pert_scale=0):
  Analysis 1: Basic metrics at SISU=0.5: peak_vel, ep_err, lat_dev
  Analysis 2: SISU=0 vs SISU=1 peak velocity (and difference)
  Analysis 3: SISU sweep {0.0, 0.25, 0.5, 0.75, 1.0} for models 1, 3, 4

Usage:
    uv run python results/2ef67ca/scripts/eval_centerout.py
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

# (index, method, pert_std, update, ratio, model_dir)
CONDITIONS = [
    (1, "std",     0, False, None,  "centerout_baseline_std_pert0"),
    (2, "std",     1, False, None,  "centerout_baseline_std_pert1"),
    (3, "std",     0, True,  0.3,   "centerout_std_update_r03_pert0"),
    (4, "std",     1, True,  0.3,   "centerout_std_update_r03_pert1"),
    (5, "std",     0, True,  0.5,   "baseline_std_update_r05_pert0"),
    (6, "APT",     0, True,  0.3,   "baseline_apt_update_r03_pert0"),
]

# Models for SISU sweep (Analysis 3): models 1, 3, 4 (indices)
SWEEP_INDICES = {1, 3, 4}

SISU_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_at_pert0(task, model, sisu: float, *, key):
    """Evaluate with pert_scale=0 at a given SISU level.

    Returns:
        km: dict with "peak_velocity", "endpoint_error", "max_lateral_deviation"
    """
    return _canonical_eval_at_pert0(task, model, sisu, key=key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading all conditions...\n")
    loaded = {}
    for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
        label = f"#{idx} {method} pert={pert_std} upd={update} r={ratio}"
        print(f"  {label} ({model_dir}) ... ", end="", flush=True)
        result = load_condition(model_dir)
        if result is None:
            print("SKIPPED — not found")
            loaded[idx] = None
        else:
            print("OK")
            loaded[idx] = result
    print()

    # =========================================================================
    # Analysis 1: Basic metrics at SISU=0.5, pert_scale=0
    # =========================================================================
    print("=" * 100)
    print("ANALYSIS 1: Basic metrics at SISU=0.5, pert_scale=0")
    print("=" * 100)

    a1 = []
    for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
        entry = loaded[idx]
        if entry is None:
            a1.append(dict(idx=idx, status="missing"))
            continue
        task, model, config = entry
        key = jr.PRNGKey(10 + idx)
        km = eval_at_pert0(task, model, sisu=0.5, key=key)
        a1.append(dict(
            idx=idx,
            method=method,
            pert_std=pert_std,
            update=update,
            ratio=ratio,
            status="ok",
            peak_vel=float(np.mean(km["peak_velocity"])),
            ep_err=float(np.mean(km["endpoint_error"])),
            lat_dev=float(np.mean(km["max_lateral_deviation"])),
        ))

    print()
    hdr = f"{'#':>2} | {'method':>5} | {'pert_std':>8} | {'loss_upd':>8} | {'ratio':>5} | {'vel(SISU=0.5)':>13} | {'ep_err':>8} | {'lat_dev':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in a1:
        if r["status"] == "missing":
            print(f"{r['idx']:>2} | MISSING")
            continue
        upd_str = "yes" if r["update"] else "no"
        ratio_str = f"{r['ratio']}" if r["ratio"] is not None else "—"
        print(
            f"{r['idx']:>2} | {r['method']:>5} | {r['pert_std']:>8.0f} | {upd_str:>8} | "
            f"{ratio_str:>5} | {r['peak_vel']:>13.4f} | {r['ep_err']:>8.4f} | {r['lat_dev']:>8.4f}"
        )
    print()

    # =========================================================================
    # Analysis 2: SISU=0 vs SISU=1 at pert_scale=0
    # =========================================================================
    print("=" * 100)
    print("ANALYSIS 2: SISU=0 vs SISU=1 peak velocity (pert_scale=0)")
    print("=" * 100)

    a2 = []
    for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
        entry = loaded[idx]
        if entry is None:
            a2.append(dict(idx=idx, status="missing"))
            continue
        task, model, config = entry
        key = jr.PRNGKey(20 + idx)
        key, k0 = jr.split(key)
        km0 = eval_at_pert0(task, model, sisu=0.0, key=k0)
        key, k1 = jr.split(key)
        km1 = eval_at_pert0(task, model, sisu=1.0, key=k1)
        vel0 = float(np.mean(km0["peak_velocity"]))
        vel1 = float(np.mean(km1["peak_velocity"]))
        a2.append(dict(
            idx=idx,
            method=method,
            pert_std=pert_std,
            update=update,
            ratio=ratio,
            status="ok",
            vel0=vel0,
            vel1=vel1,
            delta=vel1 - vel0,
        ))

    print()
    hdr2 = f"{'#':>2} | {'method':>5} | {'pert_std':>8} | {'loss_upd':>8} | {'ratio':>5} | {'vel(SISU=0)':>11} | {'vel(SISU=1)':>11} | {'Δvel':>8}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in a2:
        if r["status"] == "missing":
            print(f"{r['idx']:>2} | MISSING")
            continue
        upd_str = "yes" if r["update"] else "no"
        ratio_str = f"{r['ratio']}" if r["ratio"] is not None else "—"
        print(
            f"{r['idx']:>2} | {r['method']:>5} | {r['pert_std']:>8.0f} | {upd_str:>8} | "
            f"{ratio_str:>5} | {r['vel0']:>11.4f} | {r['vel1']:>11.4f} | {r['delta']:>+8.4f}"
        )
    print()

    # =========================================================================
    # Analysis 3: SISU sweep for models 1, 3, 4
    # =========================================================================
    print("=" * 100)
    print("ANALYSIS 3: SISU sweep {0.0, 0.25, 0.5, 0.75, 1.0} for models 1, 3, 4 (pert_scale=0)")
    print("=" * 100)

    a3 = []
    for idx, method, pert_std, update, ratio, model_dir in CONDITIONS:
        if idx not in SWEEP_INDICES:
            continue
        entry = loaded[idx]
        if entry is None:
            a3.append(dict(idx=idx, status="missing"))
            continue
        task, model, config = entry
        key = jr.PRNGKey(30 + idx)
        vels = []
        for sisu in SISU_LEVELS:
            key, k = jr.split(key)
            km = eval_at_pert0(task, model, sisu=sisu, key=k)
            vels.append(float(np.mean(km["peak_velocity"])))
        a3.append(dict(
            idx=idx,
            method=method,
            pert_std=pert_std,
            update=update,
            ratio=ratio,
            status="ok",
            vels=vels,
        ))

    print()
    sisu_headers = " | ".join(f"{'vel(SISU=' + str(s) + ')':>12}" for s in SISU_LEVELS)
    hdr3 = f"{'#':>2} | {'method':>5} | {'pert_std':>8} | {'loss_upd':>8} | {'ratio':>5} | {sisu_headers}"
    print(hdr3)
    print("-" * len(hdr3))
    for r in a3:
        if r["status"] == "missing":
            print(f"{r['idx']:>2} | MISSING")
            continue
        upd_str = "yes" if r["update"] else "no"
        ratio_str = f"{r['ratio']}" if r["ratio"] is not None else "—"
        vel_parts = " | ".join(f"{v:>12.4f}" for v in r["vels"])
        print(
            f"{r['idx']:>2} | {r['method']:>5} | {r['pert_std']:>8.0f} | {upd_str:>8} | "
            f"{ratio_str:>5} | {vel_parts}"
        )
    print()

    print("Notes:")
    print("  - ALL evaluations use pert_scale=0 (perturbation amplitude zeroed via eqx.tree_at).")
    print("  - Each model is evaluated on its own task's validation trials.")
    print("  - Peak velocity = max speed (L2 norm) after go cue, averaged over trials & replicates.")
    print("  - Center-out task: translation-invariant reaches from center to 8 targets.")


if __name__ == "__main__":
    main()
