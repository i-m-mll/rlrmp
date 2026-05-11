"""Corrected training-loss curve plot for the 6-cell anti-anticipation matrix run.

Fixes:
  1. Loads warmup_history.eqx correctly by building a proper skeleton using the
     per-cell CLI args from RUN_PLAN.md — the prior script used wrong artifact
     paths and summed raw leaf arrays (ignoring weights).
  2. Uses TermTree.aggregate() / .total to compute the TRUE weighted total loss.
  3. Uses the per-cell loss term breakdown via TermTree.flatten() for the
     per-term subplot figure.
  4. HTML only — no PNG outputs.
  5. Saves spec.json according to CLAUDE.md conventions.

Usage (from repo root):
    uv run python scripts/plot_training_loss_6cell_v2.py
"""

import argparse
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))

import equinox as eqx
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from train_minimax import build_hps
from feedbax.train import init_task_trainer_history, TaskTrainerHistory
from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476 — project-config routing
from rlrmp.modules.training.part2 import setup_task_model_pair


# ---------------------------------------------------------------------------
# Cell definitions (from RUN_PLAN.md)
# ---------------------------------------------------------------------------

CELL_LABELS = [
    "gru__jerk",
    "gru__jerk_motor_pre",
    "gru__jerk_smooth_high",
    "gru__jerk_motor_smooth_combo",
    "gru__jerk_loss_v_terminal",
    "gru__jerk_loss_historical",
]

CELL_DISPLAY_NAMES = {
    "gru__jerk": "Control (jerk only)",
    "gru__jerk_motor_pre": "Pre-go motor mask",
    "gru__jerk_smooth_high": "Hidden smoothness 1e2",
    "gru__jerk_motor_smooth_combo": "Pre-go + smooth (combo)",
    "gru__jerk_loss_v_terminal": "Var A: terminal vel",
    "gru__jerk_loss_historical": "Var B: historical shape",
}

# Per-cell CLI args that differ from the shared defaults.
# All cells share: --nn-output-jerk 1e5, --n-warmup-batches 12000,
#   --n-adversary-batches 0, --batch-size 250, --n-replicates 5, --seed 42.
CELL_EXTRA_ARGS = {
    "gru__jerk": {},
    "gru__jerk_motor_pre": {"nn_output_pre_go": 1e-2},
    "gru__jerk_smooth_high": {"nn_hidden_derivative": 1e2},
    "gru__jerk_motor_smooth_combo": {"nn_output_pre_go": 1e-2, "nn_hidden_derivative": 1e2},
    "gru__jerk_loss_v_terminal": {"effector_final_vel": 1.0, "effector_vel_late": 0.0},
    "gru__jerk_loss_historical": {
        "effector_final_vel": 1.0,
        "effector_vel_late": 0.0,
        "effector_pos_running": 0.0,
        "effector_pos_late_weight": 1.0,
        "effector_pos_late_final_scale": 6.0,
        "effector_pos_late_start_step": 0,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_with_alpha(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb + alpha to rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _make_args_namespace(label: str) -> argparse.Namespace:
    """Build an argparse.Namespace with the correct per-cell CLI flags."""
    defaults = dict(
        n_warmup_batches=N_WARMUP_BATCHES,
        n_adversary_batches=0,
        batch_size=250,
        n_replicates=N_REPLICATES,
        nn_output_jerk=1e5,
        seed=42,
        hidden_type="gru",
        sisu_gating="additive",
        loss_update_enabled=False,
        loss_update_ratio=0.5,
        # Default loss weights (from build_hps):
        effector_pos_running=1.0,
        effector_pos_late_weight=0.5,
        effector_vel_late=0.1,
        effector_final_vel=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        nn_hidden_derivative=0.0,
        nn_output_pre_go=0.0,
        nn_hidden_derivative_pre_go=0.0,
        # controller LR (needed by build_hps)
        controller_lr=1e-4,
    )
    defaults.update(CELL_EXTRA_ARGS[label])
    return argparse.Namespace(**defaults)


def load_warmup_history(artifact_dir: Path, label: str) -> TaskTrainerHistory:
    """Load warmup_history.eqx for a given cell by building the proper skeleton.

    The file is in feedbax._io.save format: JSON hyperparameters on line 1,
    then eqx.tree_serialise_leaves data starting on line 2.
    We skip the header and deserialise using the skeleton from
    init_task_trainer_history.
    """
    history_path = artifact_dir / "warmup_history.eqx"
    if not history_path.exists():
        raise FileNotFoundError(f"warmup_history.eqx not found: {history_path}")

    # Build skeleton using the per-cell args
    args = _make_args_namespace(label)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    skeleton = init_task_trainer_history(
        loss_func=pair.task.loss_func,
        n_batches=N_WARMUP_BATCHES,
        n_replicates=N_REPLICATES,
        ensembled=True,
    )

    with open(history_path, "rb") as f:
        f.readline()  # skip the JSON hyperparameters header
        history = eqx.tree_deserialise_leaves(f, skeleton)

    return history


def compute_total_loss(history: TaskTrainerHistory) -> np.ndarray:
    """Compute the TRUE weighted total training loss.

    Uses TermTree.aggregate() which respects all per-term weights.
    Returns array of shape (n_batches, n_replicates).
    """
    loss_tree = history.loss
    # aggregate() returns scalar (or broadcastable); here each leaf has shape
    # (n_batches, n_replicates) so aggregate applies weights and sums over leaves,
    # yielding shape (n_batches, n_replicates).
    total = np.array(loss_tree.aggregate(leaf_fn=lambda x: x))
    return total  # (n_batches, n_replicates)


def compute_term_losses(history: TaskTrainerHistory) -> dict[str, np.ndarray]:
    """Compute the weighted per-term losses.

    Returns dict mapping term_path -> array of shape (n_batches, n_replicates).
    Weights are applied (i.e. these are weight * raw_value, matching the
    contribution to the total).
    """
    # flatten() with apply_weights=True multiplies each leaf by its cumulative weight
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
    """Add mean + ±1σ error band traces to a plotly figure."""
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
        name=f"{display_name} +1σ",
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
        name=f"{display_name} -1σ",
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
# Cell-2 spike investigation
# ---------------------------------------------------------------------------

def investigate_cell2_spikes(history: TaskTrainerHistory, label: str) -> dict:
    """Investigate downward spikes in cell-2 (gru__jerk_motor_pre) loss traces.

    Spikes are defined as batches where at least one replicate's total loss
    drops by > 2 orders of magnitude relative to the prior running median.
    Returns a dict with investigation results.
    """
    total = compute_total_loss(history)  # (n_batches, n_replicates)
    flat = compute_term_losses(history)  # {term: (n_batches, n_replicates)}

    n_batches, n_replicates = total.shape

    results = {
        "label": label,
        "n_batches": n_batches,
        "n_replicates": n_replicates,
        "total_at_end": {
            "mean": float(total[-100:].mean()),
            "std": float(total[-100:].std()),
        },
        "spikes": [],
    }

    # Compute per-replicate "spike" threshold: drop > factor 10 vs rolling
    # median over preceding 50 batches.
    for rep_idx in range(n_replicates):
        rep_total = total[:, rep_idx]  # (n_batches,)

        # Rolling median over a 50-batch window
        window = 50
        rolling_med = np.array([
            np.median(rep_total[max(0, i - window):i + 1])
            for i in range(n_batches)
        ])

        # A spike is a batch where loss < rolling_med / 10 AND loss is at least
        # 2 orders of magnitude below the overall median.
        overall_med = np.median(rep_total)
        spike_mask = (rep_total < rolling_med / 10) & (rep_total < overall_med / 100)
        spike_batches = np.where(spike_mask)[0]

        for batch_idx in spike_batches:
            # Find which terms are most responsible
            term_at_spike = {
                k: float(v[batch_idx, rep_idx]) for k, v in flat.items()
            }
            term_at_spike_sorted = dict(
                sorted(term_at_spike.items(), key=lambda x: x[1])
            )

            # How long does the spike last?
            end_idx = batch_idx + 1
            while end_idx < n_batches and rep_total[end_idx] < rolling_med[end_idx] / 5:
                end_idx += 1
            duration = end_idx - batch_idx

            results["spikes"].append({
                "replicate": int(rep_idx),
                "batch": int(batch_idx),
                "duration_batches": int(duration),
                "total_loss_at_spike": float(rep_total[batch_idx]),
                "rolling_med_at_spike": float(rolling_med[batch_idx]),
                "drop_factor": float(rolling_med[batch_idx] / max(rep_total[batch_idx], 1e-20)),
                "terms_at_spike": term_at_spike_sorted,
            })

    return results


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

    # Bug: f485c26 — migrated from results/part2_5/runpod/anti_anticipation_loss_shape_6cell
    # to flat-by-hash layout under issue 2bc95fd.
    experiment = "2bc95fd"
    artifact_group_dir = artifact_base / experiment
    notes_dir = repo_root / "results" / experiment / "notes"
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
        print("No histories loaded — aborting.")
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

        # Report final 100-batch stats
        end_of_training_stats[label] = {
            "mean": float(mean[-100:].mean()),
            "std": float(std[-100:].mean()),
            "final_mean": float(mean[-1]),
            "final_std": float(std[-1]),
            "n_replicates": total.shape[1],
        }
        print(f"  {label}: final mean={mean[-1]:.4f}, std={std[-1]:.4f}")

        color = CELL_COLORS[i % len(CELL_COLORS)]
        display_name = CELL_DISPLAY_NAMES[label]

        add_replicate_band_traces(
            fig, x, mean, std, color, display_name, legendgroup=label
        )

    fig.update_layout(
        title="Anti-anticipation 6-cell matrix: total weighted training loss vs batch",
        xaxis_title="Training batch",
        yaxis_title="Total weighted loss (sum of weight × term)",
        width=1000,
        height=500,
        legend=dict(orientation="v", x=1.0, xanchor="left"),
        yaxis_type="log",
        xaxis_type="log",
        margin=dict(l=70, r=220, t=50, b=60),
        hovermode="x",
    )

    spec_total = {
        "figure_kind": "training_loss_total_multiline_errorbands",
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
            "error_band": "mean ± 1std",
            "yaxis_type": "log",
            "note": "Uses TermTree.aggregate() for TRUE weighted total — fixes prior bug where raw leaf sum was used.",
        },
        "end_of_training_stats": end_of_training_stats,
    }

    # Save via project-config routing (writes spec + html, creates symlink).
    total_out = save_figure(
        fig=fig, spec=spec_total,
        package="rlrmp", experiment=experiment, topic="training_loss",
        extra_packages=["rlrmp", "polars"],
    )
    print(f"Saved total-loss spec: {total_out['spec_path']}")
    print(f"Saved total-loss HTML: {total_out['render_path']}")

    # -----------------------------------------------------------------------
    # Figure 2: Per-term loss breakdown (if at least one cell loaded)
    # -----------------------------------------------------------------------
    print("\nBuilding per-term figure ...")

    # Collect all term keys across cells (union)
    all_term_keys = set()
    for label, history in histories.items():
        flat = compute_term_losses(history)
        all_term_keys.update(flat.keys())

    term_keys = sorted(all_term_keys)
    n_terms = len(term_keys)
    print(f"  Found {n_terms} loss terms: {term_keys}")

    if n_terms > 0:
        # Layout: up to 4 columns
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
                    # This cell doesn't have this term (weight=0 so it may be absent)
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

                # After adding traces, update showlegend for non-first subplot
                if not show_legend:
                    # Hide legend for all traces added in this subplot
                    for trace in fig_terms.data[-3:]:  # last 3 (mean + upper + lower)
                        trace.showlegend = False

        fig_terms.update_layout(
            title="Anti-anticipation 6-cell matrix: per-term weighted loss vs batch",
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
                "error_band": "mean ± 1std",
                "terms": term_keys,
            },
        }

        per_term_out = save_figure(
            fig=fig_terms, spec=spec_per_term,
            package="rlrmp", experiment=experiment, topic="training_loss_per_term",
            extra_packages=["rlrmp"],
        )
        print(f"Saved per-term spec: {per_term_out['spec_path']}")
        print(f"Saved per-term HTML: {per_term_out['render_path']}")

    # -----------------------------------------------------------------------
    # Cell-2 spike investigation
    # -----------------------------------------------------------------------
    print("\nInvestigating cell-2 (gru__jerk_motor_pre) spikes ...")
    cell2_label = "gru__jerk_motor_pre"
    if cell2_label in histories:
        spike_results = investigate_cell2_spikes(histories[cell2_label], cell2_label)

        # Format and save investigation notes
        notes_lines = [
            "# Cell-2 (gru__jerk_motor_pre) Spike Investigation",
            "",
            f"**Cell:** {cell2_label}",
            f"**N batches:** {spike_results['n_batches']}",
            f"**N replicates:** {spike_results['n_replicates']}",
            "",
            "## End-of-training total loss",
            "",
            f"Mean (last 100 batches): {spike_results['total_at_end']['mean']:.4e}",
            f"Std (last 100 batches): {spike_results['total_at_end']['std']:.4e}",
            "",
            f"## Detected spikes (n={len(spike_results['spikes'])})",
            "",
            "A spike is defined as: total loss drops to < 1/10 of rolling 50-batch median",
            "AND < 1/100 of the overall median.",
            "",
        ]

        if spike_results["spikes"]:
            # Table header
            notes_lines += [
                "| Replicate | Batch | Duration (batches) | Drop factor | Total at spike | Rolling median |",
                "|-----------|-------|-------------------|-------------|----------------|----------------|",
            ]
            for spike in spike_results["spikes"]:
                notes_lines.append(
                    f"| {spike['replicate']} | {spike['batch']} | {spike['duration_batches']} | "
                    f"{spike['drop_factor']:.1f}x | {spike['total_loss_at_spike']:.4e} | "
                    f"{spike['rolling_med_at_spike']:.4e} |"
                )

            notes_lines += [
                "",
                "## Per-spike loss term breakdown",
                "",
                "For each spike, the top-5 lowest weighted term values (contribution to total):",
                "",
            ]
            for spike in spike_results["spikes"]:
                notes_lines.append(
                    f"### Replicate {spike['replicate']}, batch {spike['batch']}"
                )
                terms_sorted = list(spike["terms_at_spike"].items())[:10]
                notes_lines += [
                    "",
                    "| Term | Weighted value |",
                    "|------|----------------|",
                ]
                for term, val in terms_sorted:
                    notes_lines.append(f"| {term} | {val:.4e} |")
                notes_lines.append("")

            notes_lines += [
                "## Mechanism hypothesis",
                "",
                "The `nn_output_pre_go` term (EpochMaskedLoss, weight=1e-2) rewards near-zero",
                "motor output during the pre-go epochs (epochs 0+1, before the go cue).",
                "A replicate that transiently collapses its pre-go motor activity to near-zero",
                "would show a sudden drop in the `nn_output_pre_go` term (and thus in total loss).",
                "If the replicate then reverts to anticipatory motor activity, the spike is brief.",
                "",
                "Check: if spikes are driven by `nn_output_pre_go`, then the term value at spike",
                "should be near 0, while other terms (effector_pos_running, effector_pos_late,",
                "effector_hold_pos) should be relatively unchanged.",
            ]
        else:
            notes_lines.append("No spikes detected with the defined threshold.")
            notes_lines.append("")
            notes_lines.append("### All-replicate total loss statistics")
            notes_lines.append("")
            # Show the min per replicate
            total = compute_total_loss(histories[cell2_label])
            for rep_idx in range(total.shape[1]):
                rep = total[:, rep_idx]
                notes_lines.append(
                    f"Replicate {rep_idx}: min={rep.min():.4e} at batch={int(rep.argmin())}, "
                    f"final={rep[-1]:.4e}"
                )

        notes_path = notes_dir / "cell2_spike_investigation.md"
        with open(notes_path, "w") as f:
            f.write("\n".join(notes_lines) + "\n")
        print(f"\nSaved spike investigation: {notes_path}")

        print(f"\nCell-2 spikes found: {len(spike_results['spikes'])}")
        for spike in spike_results["spikes"]:
            print(
                f"  Replicate {spike['replicate']}, batch {spike['batch']}: "
                f"{spike['drop_factor']:.1f}x drop, duration={spike['duration_batches']} batches"
            )
    else:
        print(f"  {cell2_label} not loaded — skipping spike investigation.")

    # -----------------------------------------------------------------------
    # Report: final total loss per cell
    # -----------------------------------------------------------------------
    print("\n=== Final total training loss per cell (mean ± std over replicates) ===")
    for label in CELL_LABELS:
        if label in end_of_training_stats:
            s = end_of_training_stats[label]
            print(f"  {CELL_DISPLAY_NAMES[label]:40s}: {s['final_mean']:.4e} ± {s['final_std']:.4e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
