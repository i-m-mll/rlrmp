"""Training-loss curve plot for the lit-replication 6-cell matrix (f47abb1).

Loads warmup_history.eqx correctly by building a per-cell skeleton using the
per-cell CLI args from results/f47abb1/RUN_PLAN.md.

Uses TermTree.aggregate() / .total to compute the TRUE weighted total loss.
Uses per-cell loss term breakdown via TermTree.flatten() for the per-term subplot figure.

Two figures produced (HTML only -- no PNG outputs):
  1. Total training loss overlay, all 6 cells, mean +/- 1SD error bands (log-log)
  2. Per-term loss breakdown, one subplot per term, all 6 cells overlaid

**Note on per-term figure**: with all cells having nn_hidden_derivative=0.0 (disabled),
the hidden-derivative term will be absent or near-zero. The jerk-off cells (lit__flat_nojerk,
lit__post_nojerk, lit__full_nojerk) have nn_output_jerk=0.0, so their jerk term will
also be zero or absent. This is expected and not anomalous.

**Cross-schedule absolute loss note**: total loss values are NOT comparable across
position schedule shapes. The powerlaw (t/T)^6 concentrates ~98% of position weight
in the last 30% of the trial, making the total lower for powerlaw cells than flat cells.

Usage (from repo root):
    uv run python scripts/plot_training_loss_lit_replication.py
"""

from __future__ import annotations

import argparse
from rlrmp.viz.colors import hex_to_rgba as _color_with_alpha
from pathlib import Path

from rlrmp.analysis.multi_cell_driver import (
    args_namespace,
    legacy_task_trainer_history_skeleton,
)
from rlrmp.paths import REPO_ROOT as WORKTREE  # Bug: 8404108

import equinox as eqx
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rlrmp.train.minimax_native import build_hps
from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476 -- project-config routing
from rlrmp.train.task_model import setup_task_model_pair


# ---------------------------------------------------------------------------
# Cell definitions (from results/f47abb1/RUN_PLAN.md)
# ---------------------------------------------------------------------------

CELL_LABELS = [
    "lit__flat_jerk",
    "lit__post_jerk",
    "lit__full_jerk",
    "lit__flat_nojerk",
    "lit__post_nojerk",
    "lit__full_nojerk",
]

CELL_DISPLAY_NAMES = {
    "lit__flat_jerk":    "Flat + jerk",
    "lit__post_jerk":    "Post-go PL + jerk",
    "lit__full_jerk":    "Full-trial PL + jerk",
    "lit__flat_nojerk":  "Flat, no jerk",
    "lit__post_nojerk":  "Post-go PL, no jerk",
    "lit__full_nojerk":  "Full-trial PL, no jerk",
}

# Per-cell CLI args that differ from shared defaults.
CELL_EXTRA_ARGS = {
    "lit__flat_jerk": {
        "nn_output_jerk": 1e5,
        "effector_pos_running_schedule": "flat",
        "effector_hold_pos_schedule": "flat",
    },
    "lit__post_jerk": {
        "nn_output_jerk": 1e5,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "flat",
    },
    "lit__full_jerk": {
        "nn_output_jerk": 1e5,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
    },
    "lit__flat_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_pos_running_schedule": "flat",
        "effector_hold_pos_schedule": "flat",
    },
    "lit__post_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "flat",
    },
    "lit__full_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
    },
}

# 6 visually distinct Plotly-compatible colors
CELL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

N_REPLICATES = 5
N_WARMUP_BATCHES = 12000
EXPERIMENT = "f47abb1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _make_args_namespace(label: str) -> argparse.Namespace:
    """Build argparse.Namespace with the correct per-cell CLI flags."""
    return args_namespace(
        profile="lit_replication",
        n_warmup_batches=N_WARMUP_BATCHES,
        n_replicates=N_REPLICATES,
        overrides=CELL_EXTRA_ARGS[label],
    )


def load_warmup_history(artifact_dir: Path, label: str):
    """Load warmup_history.eqx for a given cell by building the proper skeleton.

    The file is in feedbax._io.save format: JSON hyperparameters on line 1,
    then eqx.tree_serialise_leaves data starting on line 2.
    We skip the header and deserialise using the skeleton from
    init_task_trainer_history.
    """
    history_path = artifact_dir / "warmup_history.eqx"
    if not history_path.exists():
        raise FileNotFoundError(f"warmup_history.eqx not found: {history_path}")

    args = _make_args_namespace(label)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    skeleton = legacy_task_trainer_history_skeleton(
        loss_func=pair.task.loss_func,
        n_batches=N_WARMUP_BATCHES,
        n_replicates=N_REPLICATES,
        ensembled=True,
    )

    with open(history_path, "rb") as f:
        f.readline()  # skip the JSON hyperparameters header
        history = eqx.tree_deserialise_leaves(f, skeleton)

    return history


def compute_total_loss(history) -> np.ndarray:
    """Compute the TRUE weighted total training loss.

    Uses TermTree.aggregate() which respects all per-term weights.
    Returns array of shape (n_batches, n_replicates).
    """
    loss_tree = history.loss
    total = np.array(loss_tree.aggregate(leaf_fn=lambda x: x))
    return total  # (n_batches, n_replicates)


def compute_term_losses(history) -> dict[str, np.ndarray]:
    """Compute the weighted per-term losses.

    Returns dict mapping term_path -> array of shape (n_batches, n_replicates).
    Weights are applied (i.e. these are weight * raw_value).
    """
    flat = history.loss.flatten(apply_weights=True)
    return {k: np.array(v) for k, v in flat.items()}


def add_replicate_band_traces(
    fig: go.Figure,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    display_name: str,
    legendgroup: str,
    row: int = None,
    col: int = None,
):
    """Add mean +/- 1sigma error band traces to a plotly figure."""
    upper = mean + std
    lower = np.maximum(mean - std, 1e-15)

    kw = {}
    if row is not None and col is not None:
        kw = dict(row=row, col=col)

    fig.add_trace(go.Scatter(
        name=display_name,
        x=x,
        y=mean,
        mode="lines",
        line=dict(color=color, width=2),
        legendgroup=legendgroup,
        showlegend=True,
    ), **kw)

    # Upper invisible boundary
    fig.add_trace(go.Scatter(
        name=f"{display_name} +1sigma",
        x=x,
        y=upper,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=legendgroup,
    ), **kw)

    # Lower boundary with fill
    fig.add_trace(go.Scatter(
        name=f"{display_name} -1sigma",
        x=x,
        y=lower,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty",
        fillcolor=_color_with_alpha(color, 0.2),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=legendgroup,
    ), **kw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=None,
        help="Base for _artifacts dir (default: <repo-root>/_artifacts)",
    )
    args = parser.parse_args()

    repo_root = WORKTREE
    artifact_base = args.artifact_base or (repo_root / "_artifacts")

    # Bug: f47abb1 -- flat-by-hash layout under issue f47abb1.
    artifact_group_dir = artifact_base / EXPERIMENT / "runs"
    notes_dir = repo_root / "results" / EXPERIMENT / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Load all 6 cells
    histories = {}
    input_artifacts = []
    for label in CELL_LABELS:
        artifact_dir = artifact_group_dir / label
        history_path = artifact_dir / "warmup_history.eqx"
        print(f"Loading {label} from {artifact_dir} ...")
        try:
            history = load_warmup_history(artifact_dir, label)
            histories[label] = history
            input_artifacts.append({
                "path": str(history_path),
                "role": f"warmup_history:{label}",
            })
            print(f"  OK. Loss tree type: {type(history.loss).__name__}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not histories:
        print("No histories loaded -- aborting.")
        return

    # -----------------------------------------------------------------------
    # Figure 1: Total training loss, all 6 cells overlaid with error bands
    # -----------------------------------------------------------------------
    print("\nBuilding total-loss figure ...")

    fig = go.Figure()
    end_of_training_stats = {}

    for i, label in enumerate(CELL_LABELS):
        if label not in histories:
            continue

        history = histories[label]
        total = compute_total_loss(history)  # (n_batches, n_replicates)
        n_batches = total.shape[0]
        x = np.arange(n_batches)

        mean = total.mean(axis=1)
        std = total.std(axis=1)

        end_of_training_stats[label] = {
            "mean": float(mean[-100:].mean()),
            "std": float(std[-100:].mean()),
            "final_mean": float(mean[-1]),
            "final_std": float(std[-1]),
            "n_replicates": total.shape[1],
        }
        print(f"  {label}: final mean={mean[-1]:.4e}, std={std[-1]:.4e}")

        color = CELL_COLORS[i % len(CELL_COLORS)]
        display_name = CELL_DISPLAY_NAMES[label]

        add_replicate_band_traces(
            fig, x, mean, std, color, display_name, legendgroup=label
        )

    fig.update_layout(
        title=(
            "Lit-replication 6-cell matrix (f47abb1): total weighted training loss vs batch<br>"
            "<sup>Note: absolute values NOT comparable across position schedule shapes "
            "(powerlaw concentrates ~98% of weight in last 30%)</sup>"
        ),
        xaxis_title="Training batch",
        yaxis_title="Total weighted loss (sum of weight x term)",
        width=1000,
        height=500,
        legend=dict(orientation="v", x=1.0, xanchor="left"),
        yaxis_type="log",
        xaxis_type="log",
        margin=dict(l=70, r=220, t=70, b=60),
        hovermode="x",
    )

    spec_total = {
        "figure_kind": "training_loss_total_multiline_errorbands",
        "experiment": EXPERIMENT,
        "inputs": input_artifacts,
        "transform": [
            {"name": "load_warmup_history_fbx", "kwargs": {"header_lines": 1}},
            {"name": "TermTree.aggregate", "kwargs": {"leaf_fn": "identity"}},
            {"name": "mean_std_across_replicates", "kwargs": {"axis": 1}},
        ],
        "plot_kwargs": {
            "cells": CELL_LABELS,
            "n_replicates": N_REPLICATES,
            "n_warmup_batches": N_WARMUP_BATCHES,
            "error_band": "mean +/- 1std",
            "yaxis_type": "log",
            "note": (
                "Uses TermTree.aggregate() for TRUE weighted total. "
                "Cross-schedule absolute values not comparable (powerlaw ~98% weight in last 30%)."
            ),
        },
        "end_of_training_stats": end_of_training_stats,
    }

    total_out = save_figure(
        fig=fig, spec=spec_total,
        package="rlrmp", experiment=EXPERIMENT, topic="training_loss",
        extra_packages=["rlrmp", "polars"],
    )
    print(f"Saved total-loss spec: {total_out['spec_path']}")
    print(f"Saved total-loss HTML: {total_out['render_path']}")

    # -----------------------------------------------------------------------
    # Figure 2: Per-term loss breakdown
    # -----------------------------------------------------------------------
    print("\nBuilding per-term figure ...")

    all_term_keys = set()
    for label, history in histories.items():
        flat = compute_term_losses(history)
        all_term_keys.update(flat.keys())

    term_keys = sorted(all_term_keys)
    n_terms = len(term_keys)
    print(f"  Found {n_terms} loss terms: {term_keys}")

    if n_terms > 0:
        n_cols = min(4, n_terms)
        n_rows = (n_terms + n_cols - 1) // n_cols

        fig_terms = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=term_keys,
            shared_xaxes=False,
            vertical_spacing=0.12,
            horizontal_spacing=0.07,
        )

        for term_idx, term_key in enumerate(term_keys):
            row = term_idx // n_cols + 1
            col = term_idx % n_cols + 1

            for cell_i, label in enumerate(CELL_LABELS):
                if label not in histories:
                    continue

                history = histories[label]
                flat = compute_term_losses(history)

                if term_key not in flat:
                    # This cell does not have this term (weight=0 so may be absent)
                    continue

                term_vals = flat[term_key]  # (n_batches, n_replicates)
                n_batches = term_vals.shape[0]
                x = np.arange(n_batches)
                mean = term_vals.mean(axis=1)
                std = term_vals.std(axis=1)

                color = CELL_COLORS[cell_i % len(CELL_COLORS)]
                display_name = CELL_DISPLAY_NAMES[label]
                show_legend = (term_idx == 0)

                add_replicate_band_traces(
                    fig_terms, x, mean, std, color, display_name,
                    legendgroup=label,
                    row=row,
                    col=col,
                )

                if not show_legend:
                    for trace in fig_terms.data[-3:]:
                        trace.showlegend = False

        fig_terms.update_layout(
            title=(
                "Lit-replication 6-cell matrix (f47abb1): per-term weighted loss vs batch<br>"
                "<sup>nn_output_jerk absent for nojerk cells; "
                "nn_hidden_derivative=0 for all cells</sup>"
            ),
            width=300 * n_cols + 200,
            height=280 * n_rows + 100,
            yaxis_type="log",
            margin=dict(l=60, r=220, t=80, b=60),
        )
        for i in range(1, n_terms + 1):
            fig_terms.update_yaxes(type="log", row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)
            fig_terms.update_xaxes(type="log", row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

        spec_per_term = {
            "figure_kind": "training_loss_per_term_multiline_errorbands",
            "experiment": EXPERIMENT,
            "inputs": input_artifacts,
            "transform": [
                {"name": "load_warmup_history_fbx", "kwargs": {"header_lines": 1}},
                {"name": "TermTree.flatten", "kwargs": {"apply_weights": True}},
                {"name": "mean_std_across_replicates", "kwargs": {"axis": 1}},
            ],
            "plot_kwargs": {
                "cells": CELL_LABELS,
                "n_replicates": N_REPLICATES,
                "n_warmup_batches": N_WARMUP_BATCHES,
                "error_band": "mean +/- 1std",
                "terms": term_keys,
                "note": (
                    "nn_output_jerk absent/zero for nojerk cells. "
                    "nn_hidden_derivative absent/zero for all cells (weight=0). "
                    "Cross-schedule absolute loss values not comparable."
                ),
            },
        }

        per_term_out = save_figure(
            fig=fig_terms, spec=spec_per_term,
            package="rlrmp", experiment=EXPERIMENT, topic="training_loss_per_term",
            extra_packages=["rlrmp"],
        )
        print(f"Saved per-term spec: {per_term_out['spec_path']}")
        print(f"Saved per-term HTML: {per_term_out['render_path']}")

    # -----------------------------------------------------------------------
    # Report: final total loss per cell
    # -----------------------------------------------------------------------
    print("\n=== Final total training loss per cell (mean +/- std over replicates) ===")
    for label in CELL_LABELS:
        if label in end_of_training_stats:
            s = end_of_training_stats[label]
            print(f"  {CELL_DISPLAY_NAMES[label]:30s}: {s['final_mean']:.4e} +/- {s['final_std']:.4e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
