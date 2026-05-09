"""Variance + anticipation analysis for the 6-cell anti-anticipation matrix.

Computes per-(cell × replicate):
  - Peak forward velocity (m/s)
  - Hold-period / pre-go drift (mm): forward displacement during [0, go_cue)
  - Forward velocity profile over time (all 8 validation reach directions)

Then per cell:
  - Mean ± SD peak forward velocity across replicates
  - Variance ratio: SD / mean  (key decision criterion: any cell < 0.5?)
  - Mean hold drift

Produces three HTML figures:
  1. peak_velocity_distributions  — violin / box with all 6 cells (one replicate = one point)
  2. forward_velocity_profiles    — time-series per cell, one trace per replicate
  3. hold_drift_profiles          — pre-go position (forward direction) per cell

Decision criterion: variance ratio (SD/mean of peak vel across replicates) < 0.5.
Prior best (GRU/jerk baseline): 0.76.

Usage (from repo root):
    /path/to/.venv/bin/python scripts/analyse_anti_anticipation_6cell_variance.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feedbax._io import load_with_hyperparameters
from feedbax.plot.io import save_figure_with_spec

from train_minimax import build_hps
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
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

CELL_EXTRA_ARGS: dict[str, dict] = {
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

# 6 visually distinct colours
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

def _color_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _make_args_namespace(label: str) -> argparse.Namespace:
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
        effector_pos_running=1.0,
        effector_pos_late_weight=0.5,
        effector_vel_late=0.1,
        effector_final_vel=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        nn_hidden_derivative=0.0,
        nn_output_pre_go=0.0,
        nn_hidden_derivative_pre_go=0.0,
        controller_lr=1e-4,
    )
    defaults.update(CELL_EXTRA_ARGS[label])
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cell_model(label: str, artifact_base: Path):
    """Load adversarial_model.eqx for a 6-cell label.

    Returns (model, task, n_replicates).
    """
    cell_dir = artifact_base / label
    eqx_path = cell_dir / "adversarial_model.eqx"
    if not eqx_path.exists():
        raise FileNotFoundError(f"adversarial_model.eqx not found: {eqx_path}")

    args = _make_args_namespace(label)
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(42))
    task = pair.task

    model, _ = load_with_hyperparameters(
        eqx_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    # Count replicates from a weight array
    n_reps = _count_replicates(model)
    return model, task, n_reps


def _count_replicates(model) -> int:
    """Infer number of replicates from first batched leaf with ndim >= 3."""
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim >= 3:
            return int(leaf.shape[0])
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim == 2:
            # May be single replicate
            return 1
    return 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_zero_pert_trials(task, *, sisu: float = 0.5):
    """Validation trials at pert_scale=0, fixed SISU."""
    val = task.validation_trials
    n_trials = val.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val,
        jnp.zeros((n_trials,)),
    )
    trial_specs = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        trial_specs,
        jnp.full((n_trials,), sisu),
    )
    return trial_specs


def eval_ensemble(task, model, trial_specs, *, key: jax.Array, n_replicates: int):
    """Rollout the ensembled model (n_replicates leading dim).

    Returns states with shape (n_replicates, n_trials, n_steps, ...).
    """
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    def _is_rep(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    arrays, other = eqx.partition(model, _is_rep)

    def _eval_one(arr, oth, rep_key):
        m = eqx.combine(arr, oth)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(m, trial_specs, keys)

    rep_keys = jr.split(key, n_replicates)
    states = eqx.filter_vmap(_eval_one, in_axes=(0, None, 0))(arrays, other, rep_keys)
    return states  # (n_rep, n_trials, n_steps, ...)


# ---------------------------------------------------------------------------
# Kinematics computation
# ---------------------------------------------------------------------------

def compute_kinematics_per_replicate(states, trial_specs) -> dict[str, np.ndarray]:
    """Compute per-replicate kinematic metrics.

    Args:
        states: (n_rep, n_trials, n_steps, ...)
        trial_specs: TaskTrialSpec for the trials.

    Returns dict with:
        - "peak_forward_velocity": (n_rep, n_trials) m/s
        - "time_to_peak": (n_rep, n_trials) in steps
        - "forward_vel_profile": (n_rep, n_trials, n_steps) signed projection
        - "pos_forward_profile": (n_rep, n_trials, n_steps) position in reach dir
        - "hold_drift_mm": (n_rep, n_trials) pre-go forward displacement in mm
        - "go_idx": (n_trials,) step index of go cue
    """
    pos = states.mechanics.effector.pos  # (n_rep, n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel  # (n_rep, n_trials, n_steps, 2)

    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value  # (n_trials, n_steps, 2)
    goal = goal_seq[:, -1, :]  # (n_trials, 2)

    # epoch_bounds: (n_trials, 4) — columns [0, end_hold, go_cue, n_steps-1]
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,) int

    n_rep, n_trials, n_steps, _ = pos.shape
    t_arr = jnp.arange(n_steps)
    after_go = t_arr[None, None, :] >= go_idx[None, :, None]  # (1, n_trials, n_steps)
    before_go = t_arr[None, None, :] < go_idx[None, :, None]  # (1, n_trials, n_steps)

    # Reach direction: goal - pos_at_go_cue
    def _pos_at_go(pos_rep, go_arr):
        # pos_rep: (n_trials, n_steps, 2)
        return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_arr)

    pos_at_go = jax.vmap(_pos_at_go, in_axes=(0, None))(pos, go_idx)  # (n_rep, n_trials, 2)
    direction = goal[None, :, :] - pos_at_go  # (n_rep, n_trials, 2)
    d_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    d_unit = direction / jnp.maximum(d_norm, 1e-12)  # (n_rep, n_trials, 2)

    # Forward velocity profile (signed, projected onto reach axis)
    v_fwd = jnp.sum(vel * d_unit[:, :, None, :], axis=-1)  # (n_rep, n_trials, n_steps)
    v_fwd_post_go = jnp.where(after_go, v_fwd, 0.0)

    # Peak forward velocity (after go cue)
    peak_fwd = jnp.max(v_fwd_post_go, axis=-1)  # (n_rep, n_trials)

    # Time-to-peak (index)
    time_to_peak = jnp.argmax(v_fwd_post_go, axis=-1)  # (n_rep, n_trials)

    # Forward position profile (projected onto reach axis from pos_at_start=pos[:,0])
    pos_at_start = pos[:, :, 0, :]  # (n_rep, n_trials, 2) — initial position
    pos_rel = pos - pos_at_start[:, :, None, :]  # (n_rep, n_trials, n_steps, 2)
    pos_fwd = jnp.sum(pos_rel * d_unit[:, :, None, :], axis=-1)  # (n_rep, n_trials, n_steps)

    # Hold drift (mm): max forward displacement during pre-go window
    # Positive = moved toward target before go cue (anticipation)
    pos_fwd_pre_go = jnp.where(before_go, pos_fwd, -jnp.inf)
    hold_drift_m = jnp.max(pos_fwd_pre_go, axis=-1)  # (n_rep, n_trials)
    hold_drift_m = jnp.where(jnp.isinf(hold_drift_m), 0.0, hold_drift_m)
    hold_drift_mm = hold_drift_m * 1000.0

    return {
        "peak_forward_velocity": np.array(peak_fwd),       # (n_rep, n_trials)
        "time_to_peak": np.array(time_to_peak),             # (n_rep, n_trials)
        "forward_vel_profile": np.array(v_fwd),             # (n_rep, n_trials, n_steps)
        "pos_forward_profile": np.array(pos_fwd),           # (n_rep, n_trials, n_steps)
        "hold_drift_mm": np.array(hold_drift_mm),           # (n_rep, n_trials)
        "go_idx": np.array(go_idx),                         # (n_trials,)
    }


# ---------------------------------------------------------------------------
# Per-cell aggregate statistics
# ---------------------------------------------------------------------------

def compute_cell_stats(km: dict[str, np.ndarray]) -> dict:
    """Aggregate kinematics to per-replicate scalars then per-cell stats."""
    # Mean over reach directions (8 validation trials) per replicate
    peak_vel_per_rep = km["peak_forward_velocity"].mean(axis=-1)   # (n_rep,)
    hold_drift_per_rep = km["hold_drift_mm"].mean(axis=-1)          # (n_rep,)
    ttp_per_rep = km["time_to_peak"].mean(axis=-1)                   # (n_rep,)

    mean_pv = float(peak_vel_per_rep.mean())
    sd_pv = float(peak_vel_per_rep.std(ddof=1)) if len(peak_vel_per_rep) > 1 else 0.0
    variance_ratio = sd_pv / mean_pv if mean_pv > 0 else float("nan")

    return {
        "peak_vel_per_rep": peak_vel_per_rep.tolist(),
        "hold_drift_per_rep": hold_drift_per_rep.tolist(),
        "time_to_peak_per_rep": ttp_per_rep.tolist(),
        "mean_peak_velocity": mean_pv,
        "sd_peak_velocity": sd_pv,
        "variance_ratio": variance_ratio,
        "mean_hold_drift_mm": float(hold_drift_per_rep.mean()),
        "sd_hold_drift_mm": float(hold_drift_per_rep.std(ddof=1)) if len(hold_drift_per_rep) > 1 else 0.0,
        "mean_time_to_peak_steps": float(ttp_per_rep.mean()),
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def make_peak_velocity_figure(
    cell_stats: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Violin + strip plot of per-replicate peak forward velocity."""
    fig = go.Figure()
    for i, label in enumerate(CELL_LABELS):
        if label not in cell_stats:
            continue
        stats = cell_stats[label]
        pvs = stats["peak_vel_per_rep"]
        color = CELL_COLORS[i % len(CELL_COLORS)]
        display = CELL_DISPLAY_NAMES[label]
        fig.add_trace(go.Violin(
            y=pvs,
            name=display,
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.3,
            pointpos=-1.5,
            line_color=color,
            fillcolor=_color_rgba(color, 0.35),
            marker=dict(color=color, size=8),
            legendgroup=label,
            showlegend=True,
        ))

    fig.update_layout(
        title=(
            "Peak forward velocity per replicate — 6-cell anti-anticipation matrix<br>"
            "<sup>Variance ratio (SD/mean) annotated per cell. Target: < 0.5</sup>"
        ),
        yaxis_title="Peak forward velocity (m/s)",
        xaxis_title="Cell",
        width=1000,
        height=500,
        showlegend=False,
        margin=dict(l=70, r=40, t=80, b=60),
    )

    # Annotate variance ratio above each violin
    for i, label in enumerate(CELL_LABELS):
        if label not in cell_stats:
            continue
        stats = cell_stats[label]
        vr = stats["variance_ratio"]
        mean_pv = stats["mean_peak_velocity"]
        fig.add_annotation(
            x=i,
            y=max(stats["peak_vel_per_rep"]) * 1.05,
            text=f"VR={vr:.2f}",
            showarrow=False,
            font=dict(size=11, color="black"),
        )

    return fig


def make_forward_velocity_profile_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Forward velocity time series, faceted by cell (one row per cell)."""
    labels_present = [l for l in CELL_LABELS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n_cells,
        cols=1,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        v_fwd = km["forward_vel_profile"]  # (n_rep, n_trials, n_steps)
        go_idx = km["go_idx"]               # (n_trials,)
        n_rep, n_trials, n_steps = v_fwd.shape
        t = np.arange(n_steps) * dt  # seconds
        color = CELL_COLORS[CELL_LABELS.index(label) % len(CELL_COLORS)]

        # Mean over trials, then one trace per replicate
        v_mean_over_trials = v_fwd.mean(axis=1)  # (n_rep, n_steps)
        mean_go = int(go_idx.mean())

        for rep in range(n_rep):
            fig.add_trace(go.Scatter(
                x=t,
                y=v_mean_over_trials[rep],
                mode="lines",
                name=f"Rep {rep}",
                line=dict(color=_color_rgba(color, 0.7), width=1.5),
                showlegend=(row == 1),
                legendgroup=f"rep{rep}",
            ), row=row, col=1)

        # Mark go cue
        fig.add_vline(
            x=float(mean_go) * dt,
            line=dict(color="black", dash="dash", width=1),
            row=row,
            col=1,
        )

    fig.update_layout(
        title="Forward velocity profiles (mean over 8 reach directions) — 6-cell matrix",
        width=900,
        height=220 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd vel (m/s)", row=row, col=1)

    return fig


def make_hold_drift_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Pre-go forward position (hold drift) per cell, one trace per replicate."""
    labels_present = [l for l in CELL_LABELS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n_cells,
        cols=1,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        pos_fwd = km["pos_forward_profile"]  # (n_rep, n_trials, n_steps)
        go_idx = km["go_idx"]                # (n_trials,)
        n_rep, n_trials, n_steps = pos_fwd.shape
        color = CELL_COLORS[CELL_LABELS.index(label) % len(CELL_COLORS)]

        # Clip to pre-go window: steps up to max(go_idx)
        max_go = int(go_idx.max())
        t_pre = np.arange(max_go) * dt  # seconds

        # Mean over trials first
        pos_mean_over_trials = pos_fwd[:, :, :max_go].mean(axis=1)  # (n_rep, max_go)
        pos_mm = pos_mean_over_trials * 1000.0  # convert to mm

        for rep in range(n_rep):
            fig.add_trace(go.Scatter(
                x=t_pre,
                y=pos_mm[rep],
                mode="lines",
                name=f"Rep {rep}",
                line=dict(color=_color_rgba(color, 0.7), width=1.5),
                showlegend=(row == 1),
                legendgroup=f"rep{rep}",
            ), row=row, col=1)

        # Horizontal zero line
        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1), row=row, col=1)

    fig.update_layout(
        title="Pre-go forward position drift (anticipation) — 6-cell matrix",
        width=900,
        height=220 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s, pre-go)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd pos (mm)", row=row, col=1)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=None,
        help="Base directory containing the 6 cell subdirs with adversarial_model.eqx",
    )
    parser.add_argument(
        "--sisu",
        type=float,
        default=0.5,
        help="SISU level for evaluation (default: 0.5)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    artifact_base = args.artifact_base or (
        REPO_ROOT / "_artifacts" / "part2_5" / "runpod" / "anti_anticipation_loss_shape_6cell"
    )
    results_base = REPO_ROOT / "results" / "part2_5" / "runpod" / "anti_anticipation_loss_shape_6cell"

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")

    # Output paths
    notes_dir = results_base / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Figure paths: HTML to _artifacts, spec.json to results
    for fig_name in ("peak_velocity_distributions", "forward_velocity_profiles", "hold_drift_profiles"):
        (artifact_base / "figures" / fig_name).mkdir(parents=True, exist_ok=True)
        (results_base / "figures" / fig_name).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and evaluate each cell
    # -----------------------------------------------------------------------
    cell_stats: dict[str, dict] = {}
    cell_kms: dict[str, dict] = {}
    input_artifacts: list[dict] = []

    for label in CELL_LABELS:
        print(f"\n[{label}] Loading model ...", flush=True)
        try:
            model, task, n_reps = load_cell_model(label, artifact_base)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        except Exception as e:
            import traceback
            print(f"  FAILED loading: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        print(f"  Loaded. n_replicates={n_reps}")
        eqx_path = artifact_base / label / "adversarial_model.eqx"
        input_artifacts.append({"path": str(eqx_path), "role": f"adversarial_model:{label}"})

        try:
            trial_specs = build_zero_pert_trials(task, sisu=args.sisu)
            print(f"  Evaluating {n_reps} replicates on {trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]} trials ...", flush=True)
            states = eval_ensemble(task, model, trial_specs, key=jr.PRNGKey(args.eval_seed), n_replicates=n_reps)
            print(f"  Eval OK. Computing kinematics ...", flush=True)
            km = compute_kinematics_per_replicate(states, trial_specs)
            stats = compute_cell_stats(km)
            cell_kms[label] = km
            cell_stats[label] = stats
        except Exception as e:
            import traceback
            print(f"  FAILED eval/kinematics: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        print(
            f"  peak_vel: {stats['mean_peak_velocity']:.4f} ± {stats['sd_peak_velocity']:.4f} m/s  "
            f"VR={stats['variance_ratio']:.3f}  "
            f"hold_drift={stats['mean_hold_drift_mm']:.2f} ± {stats['sd_hold_drift_mm']:.2f} mm"
        )

    if not cell_stats:
        print("\nNo cells loaded — aborting.")
        return

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n--- Building figures ---")

    # Figure 1: Peak velocity distributions
    fig_pv = make_peak_velocity_figure(cell_stats)
    pv_html = artifact_base / "figures" / "peak_velocity_distributions" / "figure.html"
    fig_pv.write_html(str(pv_html))
    print(f"  Saved: {pv_html}")

    spec_pv = {
        "figure_kind": "peak_velocity_distributions_violin",
        "experiment": "anti_anticipation_loss_shape_6cell",
        "inputs": input_artifacts,
        "transform": [
            {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
            {"name": "compute_peak_forward_velocity", "kwargs": {}},
            {"name": "mean_over_trials_per_replicate", "kwargs": {}},
        ],
        "plot_kwargs": {
            "cells": CELL_LABELS,
            "n_replicates": N_REPLICATES,
            "sisu": args.sisu,
            "pert_scale": 0.0,
        },
        "cell_stats": {
            label: {
                "mean_peak_velocity": stats["mean_peak_velocity"],
                "sd_peak_velocity": stats["sd_peak_velocity"],
                "variance_ratio": stats["variance_ratio"],
                "peak_vel_per_rep": stats["peak_vel_per_rep"],
            }
            for label, stats in cell_stats.items()
        },
    }
    spec_pv_path, _ = save_figure_with_spec(
        fig_pv, spec_pv,
        results_base / "figures" / "peak_velocity_distributions",
        name="spec",
        save_render=False,
        extra_packages=["rlrmp"],
    )
    print(f"  Spec: {spec_pv_path}")

    # Figure 2: Forward velocity profiles
    if cell_kms:
        fig_fv = make_forward_velocity_profile_figure(cell_kms)
        fv_html = artifact_base / "figures" / "forward_velocity_profiles" / "figure.html"
        fig_fv.write_html(str(fv_html))
        print(f"  Saved: {fv_html}")

        spec_fv = {
            "figure_kind": "forward_velocity_profile_time_series",
            "experiment": "anti_anticipation_loss_shape_6cell",
            "inputs": input_artifacts,
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "forward_velocity_projection_onto_reach_axis", "kwargs": {}},
                {"name": "mean_over_trials", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_LABELS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "dt": 0.01,
            },
        }
        spec_fv_path, _ = save_figure_with_spec(
            fig_fv, spec_fv,
            results_base / "figures" / "forward_velocity_profiles",
            name="spec",
            save_render=False,
            extra_packages=["rlrmp"],
        )
        print(f"  Spec: {spec_fv_path}")

    # Figure 3: Hold drift profiles
    if cell_kms:
        fig_hd = make_hold_drift_figure(cell_kms)
        hd_html = artifact_base / "figures" / "hold_drift_profiles" / "figure.html"
        fig_hd.write_html(str(hd_html))
        print(f"  Saved: {hd_html}")

        spec_hd = {
            "figure_kind": "hold_drift_profile_pre_go_position",
            "experiment": "anti_anticipation_loss_shape_6cell",
            "inputs": input_artifacts,
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "forward_position_projection_onto_reach_axis", "kwargs": {}},
                {"name": "mean_over_trials", "kwargs": {}},
                {"name": "clip_to_pre_go_window", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_LABELS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "dt": 0.01,
            },
        }
        spec_hd_path, _ = save_figure_with_spec(
            fig_hd, spec_hd,
            results_base / "figures" / "hold_drift_profiles",
            name="spec",
            save_render=False,
            extra_packages=["rlrmp"],
        )
        print(f"  Spec: {spec_hd_path}")

    # -----------------------------------------------------------------------
    # Summary table + decision
    # -----------------------------------------------------------------------
    print("\n=== VARIANCE ANALYSIS SUMMARY ===\n")
    prior_best_vr = 0.76
    winner_threshold = 0.5
    winners = []

    header = f"{'Cell':42s} {'Mean PV (m/s)':>14} {'SD PV (m/s)':>12} {'VR (SD/mean)':>13} {'Hold drift (mm)':>16} {'TTP (steps)':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for label in CELL_LABELS:
        if label not in cell_stats:
            print(f"  {CELL_DISPLAY_NAMES[label]:42s} SKIPPED")
            continue
        stats = cell_stats[label]
        vr = stats["variance_ratio"]
        if vr < winner_threshold:
            winners.append(label)
        flag = " <-- WINNER" if vr < winner_threshold else (" (prior best)" if label == "gru__jerk" else "")
        print(
            f"  {CELL_DISPLAY_NAMES[label]:42s} "
            f"{stats['mean_peak_velocity']:>14.4f} "
            f"{stats['sd_peak_velocity']:>12.4f} "
            f"{vr:>13.3f} "
            f"{stats['mean_hold_drift_mm']:>16.3f} "
            f"{stats['mean_time_to_peak_steps']:>12.1f}"
            f"{flag}"
        )

    print(sep)
    print(f"\nDecision criterion: variance ratio < {winner_threshold} (prior best = {prior_best_vr})")
    if winners:
        print(f"WINNERS ({len(winners)} cells beat threshold):")
        for w in winners:
            print(f"  {CELL_DISPLAY_NAMES[w]}: VR = {cell_stats[w]['variance_ratio']:.3f}")
    else:
        best_label = min(
            (l for l in cell_stats),
            key=lambda l: cell_stats[l]["variance_ratio"],
            default=None,
        )
        if best_label:
            print(
                f"No cell met threshold. Best: {CELL_DISPLAY_NAMES[best_label]} "
                f"VR={cell_stats[best_label]['variance_ratio']:.3f} "
                f"(target was <{winner_threshold}, prior best was {prior_best_vr})"
            )
        else:
            print("No cells evaluated.")

    # -----------------------------------------------------------------------
    # Write analysis notes
    # -----------------------------------------------------------------------
    notes_path = notes_dir / "variance_analysis.md"
    lines = [
        "# Variance Analysis — 6-Cell Anti-Anticipation Matrix",
        "",
        "## Setup",
        f"- SISU: {args.sisu}",
        "- Perturbation: 0 (clean reach)",
        "- Validation trials: 8 center-out reach directions",
        "- Per-replicate metric: mean peak forward velocity across 8 directions",
        "- Variance ratio: SD(peak_vel across replicates) / mean(peak_vel across replicates)",
        "- Decision criterion: variance ratio < 0.50 (prior best = 0.76 for gru__jerk baseline)",
        "",
        "## Results Table",
        "",
        f"| Cell | Display Name | Mean PV (m/s) | SD PV (m/s) | Variance Ratio | Hold Drift (mm) | TTP (steps) | Per-rep PV |",
        f"|------|------|---------|---------|---------|---------|---------|---------|",
    ]
    for label in CELL_LABELS:
        if label not in cell_stats:
            lines.append(f"| {label} | {CELL_DISPLAY_NAMES[label]} | SKIPPED | - | - | - | - | - |")
            continue
        stats = cell_stats[label]
        pvs_str = " / ".join(f"{v:.4f}" for v in stats["peak_vel_per_rep"])
        lines.append(
            f"| {label} | {CELL_DISPLAY_NAMES[label]} "
            f"| {stats['mean_peak_velocity']:.4f} "
            f"| {stats['sd_peak_velocity']:.4f} "
            f"| {stats['variance_ratio']:.3f} "
            f"| {stats['mean_hold_drift_mm']:.3f} ± {stats['sd_hold_drift_mm']:.3f} "
            f"| {stats['mean_time_to_peak_steps']:.1f} "
            f"| {pvs_str} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        f"Threshold: variance ratio < {winner_threshold}",
        f"Prior best (gru__jerk): ~{prior_best_vr}",
        "",
    ]

    if winners:
        lines.append(f"**{len(winners)} cell(s) met the threshold:**")
        lines.append("")
        for w in winners:
            stats = cell_stats[w]
            lines.append(
                f"- **{CELL_DISPLAY_NAMES[w]}** (`{w}`): "
                f"VR = {stats['variance_ratio']:.3f}, "
                f"mean PV = {stats['mean_peak_velocity']:.4f} m/s"
            )
    else:
        best_label = min(
            (l for l in cell_stats),
            key=lambda l: cell_stats[l]["variance_ratio"],
            default=None,
        )
        if best_label:
            lines.append("**No cell met the threshold.**")
            lines.append("")
            lines.append(
                f"Best cell: **{CELL_DISPLAY_NAMES[best_label]}** (`{best_label}`) "
                f"with VR = {cell_stats[best_label]['variance_ratio']:.3f}"
            )
        else:
            lines.append("No cells evaluated.")

    lines += [
        "",
        "## Anticipation (Hold Drift)",
        "",
        "Hold drift = max forward displacement (toward target, in mm) before the go cue.",
        "Positive = anticipatory movement.",
        "",
    ]
    for label in CELL_LABELS:
        if label not in cell_stats:
            continue
        stats = cell_stats[label]
        lines.append(
            f"- {CELL_DISPLAY_NAMES[label]}: "
            f"{stats['mean_hold_drift_mm']:.3f} ± {stats['sd_hold_drift_mm']:.3f} mm"
        )

    lines += [
        "",
        "## Figures",
        "",
        "- `figures/peak_velocity_distributions/` — Violin/strip plot, one replicate per data point",
        "- `figures/forward_velocity_profiles/` — Forward velocity time series per cell",
        "- `figures/hold_drift_profiles/` — Pre-go forward position (anticipation drift)",
    ]

    with open(notes_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved analysis notes: {notes_path}")

    # Save per-cell stats as JSON for downstream use
    stats_json_path = notes_dir / "variance_analysis_data.json"
    json_data = {
        "sisu": args.sisu,
        "prior_best_variance_ratio": prior_best_vr,
        "winner_threshold": winner_threshold,
        "winners": winners,
        "cells": cell_stats,
    }
    with open(stats_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved stats JSON: {stats_json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
