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
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np

from feedbax._io import load_with_hyperparameters
from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476 — project-config routing
from feedbax.train import init_task_trainer_history
from rlrmp.eval import (
    N_REPLICATES,
    compute_kinematics,
    eval_ensemble_on_trials,
    set_sisu,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import figure_artifact_dir, figure_spec_dir, run_artifact_dir
from rlrmp.train.standard import build_hps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bulk artifact directories (gitignored, mirror of results/).
# Heavy figure renders (HTML, large PNG) go to figure_artifact_dir.
# Trained models and training histories are loaded from run_artifact_dir.
# Bug: fd64bb4 — structure migration phase 2 (role-based path helpers)
FIGURES_DIR = figure_artifact_dir("part2_5", "eval_by_sisu")
# Tracked spec directory for the spec.json + thumbnail. Bug: 0077b42 — Phase 2
# completion: figure-spec wiring via feedbax.plot.io.save_figure_with_spec.
FIGURES_SPEC_DIR = figure_spec_dir("part2_5", "eval_by_sisu")

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
# N_REPLICATES is now defined in rlrmp.eval.ensemble; imported above. Bug: 8404108.


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
# Figure-spec helpers (Bug: 0077b42)
# ---------------------------------------------------------------------------

def _condition_input_entries(loaded: dict) -> list[dict]:
    """Build ``inputs`` entries (path strings) for ``save_figure_with_spec``.

    Each loaded condition contributes its tracked ``run.json`` (or legacy
    ``config.json``) plus the bulk ``train_history.eqx`` and
    ``trained_model.eqx`` artifacts. SHA-256 digests are computed by the
    feedbax helper if absent; we list only paths here.

    Args:
        loaded: Mapping from condition name to dict containing ``"config"``
            (and the original artifact dir is recovered from the same
            ``CONDITIONS`` mapping at module scope).

    Returns:
        List of dicts with ``"path"`` keys that exist on disk.
    """
    entries: list[dict] = []
    for cond_name in loaded:
        cond_dir = CONDITIONS[cond_name]
        for fname in ("run.json", "config.json", "train_history.eqx",
                      "trained_model.eqx"):
            p = cond_dir / fname
            if p.exists():
                entries.append({"path": str(p)})
    return entries


def _save_figure(
    fig,
    name: str,
    transform_name: str,
    plot_kwargs: dict,
    inputs: list[dict],
) -> None:
    """Write a Plotly figure plus its tracked spec via project-config routing.

    Bug: f485c26, feedbax 67bf476 — migrated to ``feedbax.plot.save_figure``
    which reads rlrmp's registered ``figure_routing`` config and writes the
    spec to ``results/<exp>/figures/<topic>/spec.json`` and the heavy render
    to ``_artifacts/<exp>/figures/<topic>/figure.html``, with a relative
    symlink in the spec dir.

    Args:
        fig: A Plotly Figure.
        name: Figure topic (e.g. ``"fig_loss_curves"``); used as the
            ``topic`` segment in the routing-config templates.
        transform_name: Name of the data-transform pipeline that produced
            the figure (recorded under ``spec["transform"]``).
        plot_kwargs: The kwargs passed to the figure constructor (recorded
            under ``spec["plot_kwargs"]``).
        inputs: ``[{ "path": ... }, ...]`` listing input artifacts.
    """
    spec = {
        "figure_kind": transform_name,
        "inputs": inputs,
        "transform": [{"name": transform_name, "kwargs": {}}],
        "plot_kwargs": plot_kwargs,
    }
    save_figure(
        fig=fig, spec=spec,
        package="rlrmp", experiment="2ef67ca", topic=name,
        extra_packages=["rlrmp"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_SPEC_DIR.mkdir(parents=True, exist_ok=True)

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

    # Inputs registry shared across figures (paths to all loaded condition
    # specs + bulk artifacts; SHA-256 digests are computed by the helper).
    inputs = _condition_input_entries(loaded)
    conditions_meta = {"conditions": list(loaded.keys()), "n_replicates": N_REPLICATES}

    # --- Loss curves figure ---
    print("\nGenerating loss curves figure...")
    all_histories = {name: d["history"] for name, d in loaded.items()}
    fig_loss = make_fig_loss_curves(all_histories)
    _save_figure(
        fig_loss, "fig_loss_curves",
        transform_name="make_fig_loss_curves",
        plot_kwargs=conditions_meta,
        inputs=inputs,
    )
    print(f"  Saved render+spec: {FIGURES_DIR / 'fig_loss_curves.html'}")

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

    sisu_meta = {**conditions_meta, "sisu_levels": list(SISU_LEVELS)}

    # --- Peak velocity figure ---
    print("\nGenerating peak velocity figure...")
    fig_vel = make_fig_metric_by_sisu(
        metrics_by_cond,
        "peak_velocity",
        "Peak Parallel Velocity vs SISU Level",
        "Peak Speed (m/s)",
    )
    _save_figure(
        fig_vel, "fig_peak_velocity_by_sisu",
        transform_name="make_fig_metric_by_sisu",
        plot_kwargs={**sisu_meta, "metric": "peak_velocity"},
        inputs=inputs,
    )
    print(f"  Saved render+spec: {FIGURES_DIR / 'fig_peak_velocity_by_sisu.html'}")

    # --- Endpoint error figure ---
    print("\nGenerating endpoint error figure...")
    fig_ee = make_fig_metric_by_sisu(
        metrics_by_cond,
        "endpoint_error",
        "Endpoint Error vs SISU Level",
        "Endpoint Error (m)",
    )
    _save_figure(
        fig_ee, "fig_endpoint_error_by_sisu",
        transform_name="make_fig_metric_by_sisu",
        plot_kwargs={**sisu_meta, "metric": "endpoint_error"},
        inputs=inputs,
    )
    print(f"  Saved render+spec: {FIGURES_DIR / 'fig_endpoint_error_by_sisu.html'}")

    # --- Lateral deviation figure (no perturbation, by SISU) ---
    print("\nGenerating lateral deviation figure (no perturbation)...")
    fig_lat = make_fig_metric_by_sisu(
        metrics_by_cond,
        "max_lateral_deviation",
        "Max Lateral Deviation vs SISU Level (No Perturbation)",
        "Max Lateral Deviation (m)",
    )
    _save_figure(
        fig_lat, "fig_lateral_deviation_by_sisu",
        transform_name="make_fig_metric_by_sisu",
        plot_kwargs={**sisu_meta, "metric": "max_lateral_deviation"},
        inputs=inputs,
    )
    print(f"  Saved render+spec: {FIGURES_DIR / 'fig_lateral_deviation_by_sisu.html'}")

    print("\nAll figures saved to:", FIGURES_DIR)
    print("All specs saved to:  ", FIGURES_SPEC_DIR)


if __name__ == "__main__":
    main()
