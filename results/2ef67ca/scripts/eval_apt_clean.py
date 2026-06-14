# ruff: noqa: E402
"""Clean evaluation of APT + loss_update models at pert_scale=0.

Two analyses, both using unperturbed (pert_scale=0) evaluation:

  Analysis 1: Peak velocity at SISU=0 vs SISU=1 for all conditions,
              to cleanly measure whether SISU modulates velocity.

  Analysis 2: Peak velocity vs SISU sweep {0.0, 0.25, 0.5, 0.75, 1.0}
              for selected conditions.

Models evaluated:
  APT + loss_update:
    apt_ratio03_pert1, apt_ratio05_pert1
    apt_ratio03_pert10, apt_ratio05_pert10

  Standard + loss_update (Phase 4):
    ratio03_pert1_v4, ratio05_pert1_v4
    ratio03_pert10_v4, ratio05_pert10_v4

  Baselines:
    running_cost_standard  (pert_std=1, no loss_update)
    baseline_no_pert       (pert_std=0, no loss_update) — if present
    apt_lr001              (APT, no loss_update) — if present

Usage:
    uv run python results/2ef67ca/scripts/eval_apt_clean.py
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
from feedbax import load_with_hyperparameters

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

# (display_name, model_dir, train_pert_std, model_type)
# model_type: "baseline", "std_update", "apt_no_update", "apt_update"
ALL_CONDITIONS = [
    # Baselines
    ("baseline (std, pert_std=1)",   "running_cost_standard",  1.0,  "baseline"),
    ("baseline (no pert)",           "baseline_no_pert",        0.0,  "baseline"),
    ("APT (no update, lr=0.01)",     "apt_lr001",               1.0,  "apt_no_update"),
    # Standard + loss_update (Phase 4)
    ("std+update r=0.3 pert_std=1",  "ratio03_pert1_v4",        1.0,  "std_update"),
    ("std+update r=0.5 pert_std=1",  "ratio05_pert1_v4",        1.0,  "std_update"),
    ("std+update r=0.3 pert_std=10", "ratio03_pert10_v4",       10.0, "std_update"),
    ("std+update r=0.5 pert_std=10", "ratio05_pert10_v4",       10.0, "std_update"),
    # APT + loss_update
    ("APT+update r=0.3 pert_std=1",  "apt_ratio03_pert1",       1.0,  "apt_update"),
    ("APT+update r=0.5 pert_std=1",  "apt_ratio05_pert1",       1.0,  "apt_update"),
    ("APT+update r=0.3 pert_std=10", "apt_ratio03_pert10",      10.0, "apt_update"),
    ("APT+update r=0.5 pert_std=10", "apt_ratio05_pert10",      10.0, "apt_update"),
]

# Conditions for the SISU sweep (Analysis 2)
SISU_SWEEP_CONDITIONS = [
    "baseline (std, pert_std=1)",
    "baseline (no pert)",
    "std+update r=0.3 pert_std=10",
    "std+update r=0.5 pert_std=10",
    "APT+update r=0.3 pert_std=10",
    "APT+update r=0.5 pert_std=10",
]

SISU_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


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
    """Evaluate with pert_scale=0 (unperturbed) at a given SISU level.

    Uses the model's own task, sets pert_scale to exactly 0 via tree_at,
    so even pert_std > 0 models produce zero-amplitude gusts.

    Returns:
        km: dict with "peak_velocity", "endpoint_error", "max_lateral_deviation"
    """
    val_trials = task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    # Zero out perturbation scale regardless of what the task was trained with
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.zeros(pert_shape),
    )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading all conditions...\n")

    loaded = {}
    for display_name, model_dir, train_pert_std, model_type in ALL_CONDITIONS:
        print(f"  Loading {display_name} ({model_dir})...")
        result = load_condition(model_dir)
        if result is None:
            print("    SKIPPED — not found")
            loaded[display_name] = None
        else:
            loaded[display_name] = result
            print("    OK")

    print()

    # =========================================================================
    # Analysis 1: SISU=0 vs SISU=1 at pert_scale=0
    # =========================================================================
    print("=" * 90)
    print("ANALYSIS 1: Peak velocity at SISU=0 vs SISU=1 (pert_scale=0)")
    print("=" * 90)
    print()

    analysis1_results = []
    for display_name, model_dir, train_pert_std, model_type in ALL_CONDITIONS:
        entry = loaded[display_name]
        if entry is None:
            analysis1_results.append({
                "name": display_name,
                "train_pert_std": train_pert_std,
                "status": "missing",
            })
            continue

        task, model, config = entry
        key = jr.PRNGKey(1)

        key, k0 = jr.split(key)
        km0 = eval_at_pert0(task, model, sisu=0.0, key=k0)
        vel0 = float(np.mean(km0["peak_velocity"]))

        key, k1 = jr.split(key)
        km1 = eval_at_pert0(task, model, sisu=1.0, key=k1)
        vel1 = float(np.mean(km1["peak_velocity"]))

        delta = vel1 - vel0
        pct = 100.0 * delta / (vel0 + 1e-12)

        analysis1_results.append({
            "name": display_name,
            "train_pert_std": train_pert_std,
            "model_type": model_type,
            "vel_sisu0": vel0,
            "vel_sisu1": vel1,
            "delta_vel": delta,
            "pct_change": pct,
            "status": "ok",
        })
        print(f"  {display_name:<40}  vel@SISU=0={vel0:.3f}  vel@SISU=1={vel1:.3f}  "
              f"Δ={delta:+.3f} ({pct:+.1f}%)")

    print()
    print(f"{'Condition':<42} | {'train_pert_std':>14} | {'vel@SISU=0':>10} | "
          f"{'vel@SISU=1':>10} | {'Δvel':>8}")
    print("-" * 95)
    for r in analysis1_results:
        if r["status"] == "missing":
            print(f"  {r['name']:<40}   MISSING")
            continue
        print(f"  {r['name']:<40} | {r['train_pert_std']:>14.1f} | "
              f"{r['vel_sisu0']:>10.3f} | {r['vel_sisu1']:>10.3f} | "
              f"{r['delta_vel']:>+8.3f}")
    print()

    # =========================================================================
    # Analysis 2: SISU sweep at pert_scale=0
    # =========================================================================
    print("=" * 90)
    print("ANALYSIS 2: Peak velocity vs SISU sweep (pert_scale=0)")
    print("=" * 90)
    print()

    # Collect per-condition SISU sweep data
    sweep_results = {}
    for display_name, model_dir, train_pert_std, model_type in ALL_CONDITIONS:
        if display_name not in SISU_SWEEP_CONDITIONS:
            continue
        entry = loaded[display_name]
        if entry is None:
            sweep_results[display_name] = None
            continue

        task, model, config = entry
        key = jr.PRNGKey(2)
        vels = []
        for sisu in SISU_LEVELS:
            key, k = jr.split(key)
            km = eval_at_pert0(task, model, sisu=sisu, key=k)
            vels.append(float(np.mean(km["peak_velocity"])))
        sweep_results[display_name] = vels
        print(f"  {display_name}: " + "  ".join(f"SISU={s:.2f}->{v:.3f}"
              for s, v in zip(SISU_LEVELS, vels)))

    print()
    header_parts = [f"{'SISU=' + str(s):>10}" for s in SISU_LEVELS]
    print(f"{'Condition':<42} | " + " | ".join(header_parts))
    print("-" * (42 + 4 + 13 * len(SISU_LEVELS)))
    for display_name in SISU_SWEEP_CONDITIONS:
        vels = sweep_results.get(display_name)
        if vels is None:
            print(f"  {display_name:<40}   MISSING")
            continue
        vel_parts = [f"{v:>10.3f}" for v in vels]
        print(f"  {display_name:<40} | " + " | ".join(vel_parts))

    print()
    print("Notes:")
    print("  - ALL evaluations use pert_scale=0 (no perturbation applied during eval).")
    print("  - Each model evaluated on its own task's validation trials.")
    print("  - Perturbation amplitude zeroed via eqx.tree_at on the intervenor scale.")
    print("  - Peak velocity = max speed (L2 norm of velocity) after go cue,")
    print("    averaged over trials and replicates.")


if __name__ == "__main__":
    main()
