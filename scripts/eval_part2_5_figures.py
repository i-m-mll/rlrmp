"""Part 2.5 evaluation and figure generation script.

Loads trained models from the running_cost_standard and softmin_standard
conditions, evaluates at different SISU levels, and produces plotly figures.

Usage:
    python scripts/eval_part2_5_figures.py
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

# Add scripts/ to path for train_part2_5 import
sys.path.insert(0, str(Path(__file__).parent))

from train_part2_5 import build_hps
from feedbax._io import load_with_hyperparameters
from feedbax.train import init_task_trainer_history
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.paths import figure_artifact_dir, run_artifact_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bulk artifact directories (gitignored, mirror of results/).
# Heavy figure renders (HTML, large PNG) go to figure_artifact_dir.
# Trained models and training histories are loaded from run_artifact_dir.
# Bug: fd64bb4 — structure migration phase 2 (role-based path helpers)
FIGURES_DIR = figure_artifact_dir("part2_5", "eval_by_sisu")

CONDITIONS = {
    "running_cost": run_artifact_dir("part2_5", "running_cost__standard"),
    "softmin": run_artifact_dir("part2_5", "softmin__standard"),
    "default": run_artifact_dir("part2_5", "default__standard"),
    "combined": run_artifact_dir("part2_5", "combined__standard"),
    "CVaR_10pct": run_artifact_dir("part2_5", "running_cost__cvar"),
    "nn_1e-4": run_artifact_dir("part2_5", "running_cost__nn1e4"),
    "nn_1e-6": run_artifact_dir("part2_5", "running_cost__nn1e6"),
}

SISU_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_REPLICATES = 5


def load_condition(cond_name: str, cond_dir: Path):
    """Load config, training history, and trained model for a condition."""
    config_path = cond_dir / "config.json"
    if not config_path.exists():
        print(f"  {cond_name}: skipped (not found: {cond_dir})")
        return None
    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    # Load training history
    loss_func = pair.task.loss_func
    history_skeleton = init_task_trainer_history(
        loss_func=loss_func,
        n_batches=config["n_batches"],
        n_replicates=N_REPLICATES,
        ensembled=True,
    )
    with open(cond_dir / "train_history.eqx", "rb") as f:
        f.readline()  # skip hyperparameters line
        history = eqx.tree_deserialise_leaves(f, history_skeleton)

    # Load trained model
    trained_model, _ = load_with_hyperparameters(
        cond_dir / "trained_model.eqx",
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    return pair.task, trained_model, history, config


def set_sisu(val_trials, sisu_val: float):
    """Return a copy of val_trials with fixed SISU level."""
    n_trials = val_trials.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    new_trials = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val_trials,
        jnp.full((n_trials,), sisu_val),
    )
    new_trials = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        new_trials,
        jnp.full((n_trials,), sisu_val),
    )
    return new_trials


def eval_ensemble_on_trials(task, model, trial_specs, *, key):
    """Evaluate all N_REPLICATES on the given trial_specs.

    Uses the same partitioning strategy as feedbax's _eval_ensemble to handle
    model leaves that don't have the ensemble dimension (e.g. StateIndex.init.field).

    Returns states with leading replicate dimension: (n_replicates, n_trials, n_steps, ...).
    """
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    def _is_batched_array(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == N_REPLICATES

    models_arrays, models_other = eqx.partition(model, _is_batched_array)

    def eval_one_replicate(model_arrays, model_other, rep_key):
        rep_model = eqx.combine(model_arrays, model_other)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(rep_model, trial_specs, keys)

    rep_keys = jr.split(key, N_REPLICATES)
    states = eqx.filter_vmap(
        eval_one_replicate,
        in_axes=(0, None, 0),
    )(models_arrays, models_other, rep_keys)
    return states


def compute_kinematics(states, trial_specs):
    """Compute kinematic metrics from states.

    Arguments:
        states: Shape (n_replicates, n_trials, n_steps, ...) or (n_trials, n_steps, ...).
        trial_specs: TaskTrialSpec for the trials.

    Returns:
        Dict of arrays, each shape (n_replicates, n_trials) or (n_trials,).
    """
    pos = states.mechanics.effector.pos  # (..., n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel  # (..., n_trials, n_steps, 2)

    # Goal position: targets has shape (n_trials, n_steps, 2); take final step
    # as the endpoint target. The target key is a lambda function (WhereDict).
    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value  # (n_trials, n_steps, 2)
    goal = goal_seq[:, -1, :]  # (n_trials, 2) — final timestep target

    # Go cue step: epoch_bounds[:, 2] = start of movement epoch
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)

    # Handle optional leading replicate dim
    has_rep_dim = pos.ndim == 4  # (n_rep, n_trials, n_steps, 2)

    if has_rep_dim:
        n_rep, n_trials, n_steps, _ = pos.shape
        t = jnp.arange(n_steps)
        # after_go: (n_trials, n_steps) broadcast to (n_rep, n_trials, n_steps)
        after_go = t[None, None, :] >= go_idx[None, :, None]

        speed = jnp.linalg.norm(vel, axis=-1)  # (n_rep, n_trials, n_steps)
        masked_speed = jnp.where(after_go, speed, 0.0)
        peak_velocity = jnp.max(masked_speed, axis=-1)  # (n_rep, n_trials)

        final_pos = pos[:, :, -1, :]  # (n_rep, n_trials, 2)
        endpoint_error = jnp.linalg.norm(
            final_pos - goal[None, :, :], axis=-1
        )  # (n_rep, n_trials)

        # Lateral deviation: per trial, get initial pos at go cue
        def get_init_pos_rep(pos_rep, go_idx_arr):
            # pos_rep: (n_trials, n_steps, 2), go_idx_arr: (n_trials,)
            return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_idx_arr)

        init_pos = jax.vmap(get_init_pos_rep, in_axes=(0, None))(
            pos, go_idx
        )  # (n_rep, n_trials, 2)

        direction = goal[None, :, :] - init_pos  # (n_rep, n_trials, 2)
        direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        direction_unit = direction / jnp.maximum(direction_norm, 1e-12)

        displacement = pos - init_pos[:, :, None, :]  # (n_rep, n_trials, n_steps, 2)
        along = jnp.sum(
            displacement * direction_unit[:, :, None, :], axis=-1, keepdims=True
        )
        lateral = displacement - along * direction_unit[:, :, None, :]
        lateral_dist = jnp.linalg.norm(lateral, axis=-1)  # (n_rep, n_trials, n_steps)
        masked_lateral = jnp.where(after_go, lateral_dist, 0.0)
        max_lateral_deviation = jnp.max(masked_lateral, axis=-1)  # (n_rep, n_trials)

    else:
        n_trials, n_steps, _ = pos.shape
        t = jnp.arange(n_steps)
        after_go = t[None, :] >= go_idx[:, None]

        speed = jnp.linalg.norm(vel, axis=-1)
        masked_speed = jnp.where(after_go, speed, 0.0)
        peak_velocity = jnp.max(masked_speed, axis=-1)

        final_pos = pos[:, -1, :]
        endpoint_error = jnp.linalg.norm(final_pos - goal, axis=-1)

        init_pos = jax.vmap(lambda p, idx: p[idx])(pos, go_idx)
        direction = goal - init_pos
        direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        direction_unit = direction / jnp.maximum(direction_norm, 1e-12)
        displacement = pos - init_pos[:, None, :]
        along = jnp.sum(displacement * direction_unit[:, None, :], axis=-1, keepdims=True)
        lateral = displacement - along * direction_unit[:, None, :]
        lateral_dist = jnp.linalg.norm(lateral, axis=-1)
        masked_lateral = jnp.where(after_go, lateral_dist, 0.0)
        max_lateral_deviation = jnp.max(masked_lateral, axis=-1)

    return {
        "peak_velocity": np.array(peak_velocity),
        "endpoint_error": np.array(endpoint_error),
        "max_lateral_deviation": np.array(max_lateral_deviation),
    }


def evaluate_at_sisu(task, model, sisu_val: float, *, key):
    """Evaluate all replicates at a fixed SISU level."""
    val_trials = task.validation_trials
    modified_trials = set_sisu(val_trials, sisu_val)
    states = eval_ensemble_on_trials(task, model, modified_trials, key=key)
    metrics = compute_kinematics(states, modified_trials)
    return metrics


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _collect_leaf_totals(term_tree, weight_multiplier=1.0):
    """Recursively collect weighted sum of all leaf values in a TermTree.

    Handles branch nodes (value=None) by recursing into children.
    Returns a numpy array of shape (n_batches, n_rep) or zeros if all terms have None values.
    """
    from feedbax.loss import TermTree

    if term_tree.value is not None:
        # Leaf node
        val = np.array(term_tree.value)
        w = float(term_tree.weight) * weight_multiplier
        return w * val

    # Branch node: recurse into children
    total = None
    for child in term_tree.children:
        child_total = _collect_leaf_totals(child, weight_multiplier * float(term_tree.weight))
        if child_total is not None:
            total = child_total if total is None else total + child_total

    return total


def make_fig_loss_curves(all_histories: dict) -> "plotly.graph_objs.Figure":
    """Training loss curves for each condition."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_conds = len(all_histories)
    fig = make_subplots(
        rows=1, cols=n_conds,
        subplot_titles=list(all_histories.keys()),
        shared_yaxes=False,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for col, (cond_name, history) in enumerate(all_histories.items(), 1):
        loss = history.loss
        # Compute weighted total loss as sum over all leaf terms
        total = _collect_leaf_totals(loss)
        if total is None:
            print(f"  Warning: no leaf loss values found for condition {cond_name}")
            continue

        n_batches, n_rep = total.shape
        x = np.arange(n_batches)

        mean_loss = total.mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=x, y=mean_loss,
                mode="lines",
                name=f"{cond_name} (mean)",
                line=dict(color=colors[0], width=2),
                legendgroup=cond_name,
            ),
            row=1, col=col,
        )
        for rep in range(n_rep):
            fig.add_trace(
                go.Scatter(
                    x=x, y=total[:, rep],
                    mode="lines",
                    name=f"{cond_name} rep {rep}",
                    line=dict(color=colors[rep % len(colors)], width=1, dash="dot"),
                    opacity=0.4,
                    legendgroup=cond_name,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )

        fig.update_xaxes(title_text="Batch", row=1, col=col)
        fig.update_yaxes(title_text="Loss", row=1, col=col)

    fig.update_layout(
        title="Part 2.5: Training Loss Curves",
        height=400,
        width=900 * n_conds,
    )
    return fig


def make_fig_metric_by_sisu(
    metrics_by_cond: dict,
    metric_key: str,
    title: str,
    ylabel: str,
) -> "plotly.graph_objs.Figure":
    """Line plot of a kinematic metric vs SISU level, per condition."""
    import plotly.graph_objects as go

    fig = go.Figure()
    cond_colors = {
        "running_cost": "#1f77b4", "softmin": "#ff7f0e",
        "default": "#2ca02c", "combined": "#d62728",
        "CVaR_10pct": "#9467bd", "nn_1e-4": "#8c564b", "nn_1e-6": "#e377c2",
    }
    dash_styles = ["solid", "dot", "dash", "longdash", "dashdot"]

    for cond_name, sisu_metrics in metrics_by_cond.items():
        color = cond_colors.get(cond_name, "#333333")
        # sisu_metrics: dict[sisu_val -> dict[metric_key -> array(n_rep, n_trials)]]
        sisu_vals = sorted(sisu_metrics.keys())

        # Per-replicate means across trials
        # metric arr: (n_rep, n_trials) -> mean over trials -> (n_rep,)
        rep_means = []
        for sv in sisu_vals:
            arr = sisu_metrics[sv][metric_key]  # (n_rep, n_trials)
            rep_means.append(arr.mean(axis=-1))  # (n_rep,)
        rep_means = np.array(rep_means)  # (n_sisu, n_rep)

        # Overall mean line
        overall_mean = rep_means.mean(axis=1)  # (n_sisu,)
        fig.add_trace(
            go.Scatter(
                x=sisu_vals, y=overall_mean,
                mode="lines+markers",
                name=f"{cond_name} (mean)",
                line=dict(color=color, width=2),
                legendgroup=cond_name,
            )
        )
        # Individual replicate lines
        n_rep = rep_means.shape[1]
        for rep in range(n_rep):
            fig.add_trace(
                go.Scatter(
                    x=sisu_vals, y=rep_means[:, rep],
                    mode="lines",
                    name=f"{cond_name} rep {rep}",
                    line=dict(color=color, width=1, dash=dash_styles[rep % len(dash_styles)]),
                    opacity=0.35,
                    legendgroup=cond_name,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="SISU Level",
        yaxis_title=ylabel,
        height=500,
        width=700,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading conditions...")
    loaded = {}
    for cond_name, cond_dir in CONDITIONS.items():
        print(f"  Loading {cond_name} from {cond_dir} ...")
        try:
            result = load_condition(cond_name, cond_dir)
            if result is None:
                continue
            task, model, history, config = result
            loaded[cond_name] = dict(task=task, model=model, history=history, config=config)
        except Exception as e:
            print(f"  {cond_name}: FAILED to load — {e}")
            continue

    # --- Loss curves figure ---
    print("\nGenerating loss curves figure...")
    all_histories = {name: d["history"] for name, d in loaded.items()}
    fig_loss = make_fig_loss_curves(all_histories)
    fig_loss.write_html(str(FIGURES_DIR / "fig_loss_curves.html"))
    print(f"  Saved: {FIGURES_DIR / 'fig_loss_curves.html'}")

    # --- Evaluate at SISU levels ---
    print("\nEvaluating at SISU levels (no evaluation perturbation)...")
    metrics_by_cond = {}
    for cond_name, d in loaded.items():
        task = d["task"]
        model = d["model"]
        metrics_by_sisu = {}
        for i, sisu_val in enumerate(SISU_LEVELS):
            print(f"  {cond_name}: SISU={sisu_val:.2f}")
            key = jr.fold_in(jr.PRNGKey(99), int(i * 100 + hash(cond_name) % 1000))
            metrics = evaluate_at_sisu(task, model, sisu_val, key=key)
            metrics_by_sisu[sisu_val] = metrics
        metrics_by_cond[cond_name] = metrics_by_sisu

    # --- Peak velocity figure ---
    print("\nGenerating peak velocity figure...")
    fig_vel = make_fig_metric_by_sisu(
        metrics_by_cond,
        "peak_velocity",
        "Peak Parallel Velocity vs SISU Level",
        "Peak Speed (m/s)",
    )
    fig_vel.write_html(str(FIGURES_DIR / "fig_peak_velocity_by_sisu.html"))
    print(f"  Saved: {FIGURES_DIR / 'fig_peak_velocity_by_sisu.html'}")

    # --- Endpoint error figure ---
    print("\nGenerating endpoint error figure...")
    fig_ee = make_fig_metric_by_sisu(
        metrics_by_cond,
        "endpoint_error",
        "Endpoint Error vs SISU Level",
        "Endpoint Error (m)",
    )
    fig_ee.write_html(str(FIGURES_DIR / "fig_endpoint_error_by_sisu.html"))
    print(f"  Saved: {FIGURES_DIR / 'fig_endpoint_error_by_sisu.html'}")

    # --- Lateral deviation figure (no perturbation, by SISU) ---
    print("\nGenerating lateral deviation figure (no perturbation)...")
    fig_lat = make_fig_metric_by_sisu(
        metrics_by_cond,
        "max_lateral_deviation",
        "Max Lateral Deviation vs SISU Level (No Perturbation)",
        "Max Lateral Deviation (m)",
    )
    fig_lat.write_html(str(FIGURES_DIR / "fig_lateral_deviation_by_sisu.html"))
    print(f"  Saved: {FIGURES_DIR / 'fig_lateral_deviation_by_sisu.html'}")

    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
