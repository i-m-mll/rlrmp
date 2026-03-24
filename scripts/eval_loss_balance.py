"""Evaluate the 4 loss-balance experiment models and measure the SISU → velocity effect.

Key question: Does balancing control cost against position cost (via adaptive nn_output
weight) create room for SISU to modulate velocity?

Models evaluated:
  ratio03_pert1  — target_ratio=0.3, pert_std=1.0
  ratio05_pert1  — target_ratio=0.5, pert_std=1.0
  ratio03_pert10 — target_ratio=0.3, pert_std=10.0
  ratio05_pert10 — target_ratio=0.5, pert_std=10.0

Baseline: running_cost_standard (loss_update disabled, pert_std=1.0)

Metrics reported per condition:
  - Final training loss (median over replicates, last batch)
  - Endpoint error at SISU=0.5, pert_scale=0
  - Peak forward velocity at SISU=0.5, pert_scale=0
  - SISU 0 vs SISU 1 peak velocity difference at pert_scale=0.5
  - Lateral deviation under perturbation at pert_scale=1.0, SISU=0.5
  - Final nn_output weight (what the adaptive update converged to)

Usage:
    uv run python scripts/eval_loss_balance.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np

WORKTREE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))

from train_part2_5 import build_hps  # noqa: E402
from eval_part2_5_figures import (  # noqa: E402
    eval_ensemble_on_trials,
    compute_kinematics,
    set_sisu,
    N_REPLICATES,
)
from feedbax._io import load_with_hyperparameters  # noqa: E402
from feedbax.train import init_task_trainer_history  # noqa: E402
from rlrmp.modules.training.part2 import setup_task_model_pair  # noqa: E402
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL  # noqa: E402

RESULTS_BASE = WORKTREE / "results" / "part2_5"
MODELS_BASE = RESULTS_BASE / "models"

# Conditions: (display_name, model_dir, has_loss_update)
CONDITIONS = [
    ("running_cost_standard", "running_cost_standard", False),
    ("ratio03_pert1",  "ratio03_pert1",  True),
    ("ratio05_pert1",  "ratio05_pert1",  True),
    ("ratio03_pert10", "ratio03_pert10", True),
    ("ratio05_pert10", "ratio05_pert10", True),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_condition(model_dir_name: str):
    """Load task, model, and training history for a single condition.

    Returns:
        Tuple of (task, model, history, config) or None if not found.
    """
    cond_dir = MODELS_BASE / model_dir_name
    config_path = cond_dir / "config.json"
    if not config_path.exists():
        print(f"  {model_dir_name}: skipped — not found at {cond_dir}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    n_batches = config.get("n_batches", 10000)
    loss_func = pair.task.loss_func
    history_skeleton = init_task_trainer_history(
        loss_func=loss_func,
        n_batches=n_batches,
        n_replicates=N_REPLICATES,
        ensembled=True,
    )
    history_path = cond_dir / "train_history.eqx"

    def _float_to_array(x):
        if isinstance(x, (float, int)) and not isinstance(x, bool):
            return jnp.array(x)
        return x

    history_skeleton_arr = jt.map(_float_to_array, history_skeleton)

    def _multi_array_filter_spec(file, x):
        if isinstance(x, jax.Array) and x.size > 1:
            return jnp.load(file)
        elif isinstance(x, np.ndarray) and x.size > 1:
            return np.load(file)
        else:
            return x

    # Count skeleton leaves vs disk arrays to choose the right loading approach.
    # Older saves (e.g. running_cost_standard) skip scalar leaves and only save multi-element
    # arrays, so disk count < skeleton count. Newer saves (ratio conditions with loss_update)
    # save every leaf, so disk count == skeleton count. We try plain deserialization first.
    skeleton_leaves = jt.leaves(history_skeleton_arr)
    disk_arrays = []
    with open(history_path, "rb") as f:
        f.readline()  # skip header
        while True:
            try:
                disk_arrays.append(jnp.load(f))
            except Exception:
                break

    if len(disk_arrays) == len(skeleton_leaves):
        # New-style save: disk and skeleton sizes match; load directly.
        disk_leaves_iter = iter(disk_arrays)
        def _direct_filter_spec(file, x):  # noqa: E306
            return next(disk_leaves_iter)

        with open(history_path, "rb") as f:
            f.readline()  # skip header
            history = eqx.tree_deserialise_leaves(f, history_skeleton_arr)
    else:
        # Old-style save: only multi-element arrays stored; use filter_spec to skip scalars.
        with open(history_path, "rb") as f:
            f.readline()  # skip header
            history = eqx.tree_deserialise_leaves(
                f, history_skeleton_arr, filter_spec=_multi_array_filter_spec
            )

    trained_model, _ = load_with_hyperparameters(
        cond_dir / "trained_model.eqx",
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    return pair.task, trained_model, history, config


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_at_sisu_pert(task, model, sisu: float, pert_scale: float, *, key, ref_task=None):
    """Evaluate model at fixed SISU and perturbation scale.

    Args:
        task: The model's own task.
        model: Trained model.
        sisu: SISU level [0, 1].
        pert_scale: Perturbation scale factor; 0 = no perturbation.
        key: JAX PRNGKey.
        ref_task: Reference task for trial specs (used when pert_scale > 0 and
            the model's task has pert_std=0). Falls back to task if None.

    Returns:
        states: (n_rep, n_trials, n_steps, ...)
        trial_specs: modified trial specs used.
    """
    source_task = (ref_task if (ref_task is not None and pert_scale > 0) else task)
    val_trials = source_task.validation_trials
    # Set SISU
    trial_specs = set_sisu(val_trials, sisu)
    # Set perturbation scale
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.full(trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape, pert_scale),
    )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return states, trial_specs


# ---------------------------------------------------------------------------
# Loss history helpers
# ---------------------------------------------------------------------------


def get_final_loss(history, n_last: int = 100) -> float:
    """Compute mean total loss over the last n_last batches (median over replicates)."""
    total = None
    # history.losses is a TermTree; sum all weighted terms by recursing leaves
    def _collect(node):
        nonlocal total
        if node.value is not None:
            # value shape: (n_batches, n_replicates)
            v = np.array(node.value)
            w = float(node.weight)
            contrib = w * v
            return contrib
        # branch node
        child_total = None
        for child in node.children:
            ct = _collect(child)
            if ct is not None:
                child_total = ct if child_total is None else child_total + ct
        if child_total is not None:
            return float(node.weight) * child_total
        return None

    total_curve = _collect(history.loss)
    if total_curve is None:
        return float("nan")
    # total_curve: (n_batches, n_replicates)
    # Use last n_last batches, take mean over batches then median over replicates
    tail = total_curve[-n_last:]  # (n_last, n_replicates)
    return float(np.median(tail.mean(axis=0)))


def get_final_nn_output_weight(history):
    """Extract the final nn_output weight from the training history.

    For adaptive loss update conditions, the weight drifts during training
    and the history records its evolution. We want the value at the end.

    Returns:
        final_weight: float (the weight at the final batch, median over reps),
            or None if not found in the history.
    """
    # The history has a `loss_func` attribute if recorded via feedbax's history mechanism.
    # The nn_output weight is stored as a leaf in history.loss_func.weights.nn_output or similar.
    # We look for it by traversing the tree.
    try:
        # First try: history.loss_weights is an array (n_batches, n_replicates) stored
        # alongside history.losses as a separate field.
        lw = history.loss_weights
        # lw should be a dict-like structure; try to find nn_output key
        if hasattr(lw, "nn_output"):
            arr = np.array(lw.nn_output)  # (n_batches, n_replicates)
            return float(np.median(arr[-1]))
        if isinstance(lw, dict) and "nn_output" in lw:
            arr = np.array(lw["nn_output"])
            return float(np.median(arr[-1]))
    except AttributeError:
        pass

    # Fallback: look in the history loss TermTree for a node whose label is "nn_output"
    try:
        nn_node = history.loss["nn_output"]
        if nn_node is not None and nn_node.value is not None:
            # value is (n_batches, n_replicates); weight is the adaptive weight
            w = float(nn_node.weight)
            return w
    except (KeyError, TypeError, AttributeError):
        pass

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading reference task (running_cost_standard) for perturbation trials...")
    ref_result = load_condition("running_cost_standard")
    if ref_result is None:
        raise RuntimeError("Reference task (running_cost_standard) not found.")
    ref_task, _, _, _ = ref_result
    print("  OK\n")

    results = []

    for display_name, model_dir, has_loss_update in CONDITIONS:
        print(f"Evaluating {display_name}...")
        result = load_condition(model_dir)
        if result is None:
            results.append({"name": display_name, "status": "missing"})
            continue

        task, model, history, config = result
        key = jr.PRNGKey(0)

        # --- Final training loss ---
        final_loss = get_final_loss(history)

        # --- nn_output weight at end of training ---
        nn_weight = get_final_nn_output_weight(history)

        # --- Unperturbed baseline at SISU=0.5 ---
        key, k1 = jr.split(key)
        states_base, specs_base = eval_at_sisu_pert(task, model, sisu=0.5, pert_scale=0.0, key=k1)
        km_base = compute_kinematics(states_base, specs_base)
        # Flatten replicate and trial dims
        ep_err = float(np.mean(km_base["endpoint_error"]))
        peak_vel = float(np.mean(km_base["peak_velocity"]))

        # --- SISU 0 vs SISU 1 at pert_scale=0.5 ---
        key, k2 = jr.split(key)
        states_s0, specs_s0 = eval_at_sisu_pert(
            task, model, sisu=0.0, pert_scale=0.5, key=k2, ref_task=ref_task
        )
        key, k3 = jr.split(key)
        states_s1, specs_s1 = eval_at_sisu_pert(
            task, model, sisu=1.0, pert_scale=0.5, key=k3, ref_task=ref_task
        )
        km_s0 = compute_kinematics(states_s0, specs_s0)
        km_s1 = compute_kinematics(states_s1, specs_s1)
        vel_sisu0 = float(np.mean(km_s0["peak_velocity"]))
        vel_sisu1 = float(np.mean(km_s1["peak_velocity"]))
        vel_change_pct = 100.0 * (vel_sisu1 - vel_sisu0) / (vel_sisu0 + 1e-12)

        # --- Lateral deviation under perturbation at SISU=0.5, pert_scale=1.0 ---
        key, k4 = jr.split(key)
        states_lat, specs_lat = eval_at_sisu_pert(
            task, model, sisu=0.5, pert_scale=1.0, key=k4, ref_task=ref_task
        )
        km_lat = compute_kinematics(states_lat, specs_lat)
        lat_dev = float(np.mean(km_lat["max_lateral_deviation"]))

        results.append({
            "name": display_name,
            "status": "ok",
            "final_loss": final_loss,
            "nn_output_weight": nn_weight,
            "endpoint_error": ep_err,
            "peak_velocity": peak_vel,
            "vel_sisu0": vel_sisu0,
            "vel_sisu1": vel_sisu1,
            "vel_change_pct": vel_change_pct,
            "lat_dev_pert1": lat_dev,
        })
        print(f"  loss={final_loss:.4f}  ep_err={ep_err:.4f}  peak_vel={peak_vel:.3f}  "
              f"SISU 0→1 vel: {vel_change_pct:+.1f}%  lat_dev={lat_dev:.4f}  "
              f"nn_w={nn_weight}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 110)
    print(f"{'Condition':<22} | {'Loss':>7} | {'Ep err':>7} | {'Peak vel':>8} | "
          f"{'vel@SISU0':>9} | {'vel@SISU1':>9} | {'SISU 0→1 Δvel':>13} | "
          f"{'Lat dev(×1)':>11} | {'nn_output_w':>12}")
    print("-" * 110)
    for r in results:
        if r["status"] == "missing":
            print(f"  {r['name']:<20}  MISSING")
            continue
        nn_w = f"{r['nn_output_weight']:.2e}" if r["nn_output_weight"] is not None else "    N/A"
        print(
            f"  {r['name']:<20}| {r['final_loss']:>7.4f} | {r['endpoint_error']:>7.4f} | "
            f"{r['peak_velocity']:>8.3f} | {r['vel_sisu0']:>9.3f} | {r['vel_sisu1']:>9.3f} | "
            f"{r['vel_change_pct']:>+12.1f}% | {r['lat_dev_pert1']:>11.4f} | {nn_w:>12}"
        )
    print("=" * 110)
    print()
    print("Notes:")
    print("  - Unperturbed metrics (ep_err, peak_vel) evaluated at SISU=0.5, pert_scale=0")
    print("  - SISU comparison at pert_scale=0.5 using ref task (running_cost_standard) trials")
    print("  - Lateral deviation at SISU=0.5, pert_scale=1.0 using ref task trials")
    print("  - nn_output_weight: final value of the adaptive control penalty weight")
    print("    (for running_cost_standard this is fixed at the config value 1e-5)")


if __name__ == "__main__":
    main()
