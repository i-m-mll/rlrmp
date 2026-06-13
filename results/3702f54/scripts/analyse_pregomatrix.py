# TODO: relocate to results/3702f54/scripts/ — per CLAUDE.md script-placement convention
"""Pre-go motor mask follow-up matrix analysis (3702f54).

Tests whether (a) scaling position weight 10x or (b) the `--nn-output-pre-go` lever
can suppress the residual ~2-3 mm pre-go anticipation observed in f47abb1's leading
no-jerk powerlaw cells (`lit__post_nojerk`, `lit__full_nojerk`).

Bug: 06f7faf — go-cue alignment fix. Forward velocity profiles, hold drift
profiles, and the within-cell vel-RMSE metric are now computed on
per-trial go-cue-aligned profiles using `rlrmp.analysis.trial_alignment`.

Loads 10 models:
  - 8 new cells from `_artifacts/3702f54/runs/<cell>/warmup_model.eqx`
  - 2 baseline anchors from `_artifacts/f47abb1/runs/{lit__post_nojerk,lit__full_nojerk}/warmup_model.eqx`

Computes per-(cell x replicate):
  - Within-cell pairwise velocity-RMSE (m/s, absolute) -- primary frame
  - Hold drift mm (RMS forward effector position during hold period)
  - Pre-go forward drift specifically over the [-200ms, 0] pre-go window
  - Peak forward velocity (m/s)
  - Time-to-peak velocity (steps after go)

Produces 5 HTML figures via `feedbax.plot.save_figure`:
  1. forward_velocity_profiles
  2. hold_drift_profiles
  3. peak_velocity_distributions
  4. summary_metrics
  5. training_loss_per_term

Baselines from f47abb1 (`lit__post_nojerk`, `lit__full_nojerk`) are placed at the TOP
of cell ordering / first columns for direct visual comparison.

For cells where n_adversary_batches=0 the canonical final model is `warmup_model.eqx`
(Bug: a517040). Bug: f47abb1.

Usage (from repo root):
    .venv/bin/python scripts/analyse_pregomatrix.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from feedbax import load_with_hyperparameters
from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476 -- project-config routing
from feedbax.train import TaskTrainerHistory, init_task_trainer_history
from plotly.subplots import make_subplots

from rlrmp.analysis.trial_alignment import (
    align_trials,
    pooled_trial_mean_with_band,
    replicate_mean_curves,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.paths import REPO_ROOT  # Bug: 8404108 — was __file__-relative
from rlrmp.train.minimax import build_hps
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz import profile_comparison_grid

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

# Baselines at TOP for direct visual comparison (per user request).
BASELINE_CELLS = [
    "lit__post_nojerk",
    "lit__full_nojerk",
]

NEW_CELLS = [
    "post_go_pl__pos10",
    "full_trial_pl__pos10",
    "full_trial_pl__prego_1e-3",
    "full_trial_pl__prego_5e-2",
    "full_trial_pl__prego_1",
    "full_trial_pl__pos10_prego_1e-3",
    "full_trial_pl__pos10_prego_5e-2",
    "full_trial_pl__pos10_prego_1",
]

CELL_LABELS = BASELINE_CELLS + NEW_CELLS

CELL_DISPLAY_NAMES = {
    "lit__post_nojerk":                "[baseline] Post-go PL, no jerk (f47abb1)",
    "lit__full_nojerk":                "[baseline] Full-trial PL, no jerk (f47abb1)",
    "post_go_pl__pos10":               "Post-go PL, pos10",
    "full_trial_pl__pos10":            "Full-trial PL, pos10",
    "full_trial_pl__prego_1e-3":       "Full-trial PL, prego=1e-3",
    "full_trial_pl__prego_5e-2":       "Full-trial PL, prego=5e-2",
    "full_trial_pl__prego_1":          "Full-trial PL, prego=1.0",
    "full_trial_pl__pos10_prego_1e-3": "Full-trial PL, pos10, prego=1e-3",
    "full_trial_pl__pos10_prego_5e-2": "Full-trial PL, pos10, prego=5e-2",
    "full_trial_pl__pos10_prego_1":    "Full-trial PL, pos10, prego=1.0",
}

# Per-cell CLI args that differ from shared defaults. Mirrors the run.json specs.
CELL_EXTRA_ARGS: dict[str, dict] = {
    # f47abb1 baselines
    "lit__post_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "flat",
        "nn_output_pre_go": 0.0,
    },
    "lit__full_nojerk": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 0.0,
    },
    # New 3702f54 cells
    "post_go_pl__pos10": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "flat",
        "nn_output_pre_go": 0.0,
    },
    "full_trial_pl__pos10": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 0.0,
    },
    "full_trial_pl__prego_1e-3": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1e-3,
    },
    "full_trial_pl__prego_5e-2": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 5e-2,
    },
    "full_trial_pl__prego_1": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 1.0,
        "effector_pos_running": 1.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1.0,
    },
    "full_trial_pl__pos10_prego_1e-3": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1e-3,
    },
    "full_trial_pl__pos10_prego_5e-2": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 5e-2,
    },
    "full_trial_pl__pos10_prego_1": {
        "nn_output_jerk": 0.0,
        "effector_hold_pos": 10.0,
        "effector_pos_running": 10.0,
        "effector_pos_running_schedule": "powerlaw",
        "effector_hold_pos_schedule": "powerlaw",
        "nn_output_pre_go": 1.0,
    },
}

# Cells live in different artifact experiment dirs.
CELL_ARTIFACT_EXPERIMENT = {
    "lit__post_nojerk": "f47abb1",
    "lit__full_nojerk": "f47abb1",
    **{c: "3702f54" for c in NEW_CELLS},
}

# Color palette: baselines in grey shades, then qualitative palette for the matrix.
CELL_COLORS = {
    "lit__post_nojerk":                "#7f7f7f",  # grey
    "lit__full_nojerk":                "#4d4d4d",  # dark grey
    "post_go_pl__pos10":               "#1f77b4",  # blue
    "full_trial_pl__pos10":            "#17becf",  # cyan
    "full_trial_pl__prego_1e-3":       "#ffbb78",  # light orange
    "full_trial_pl__prego_5e-2":       "#ff7f0e",  # orange
    "full_trial_pl__prego_1":          "#d62728",  # red
    "full_trial_pl__pos10_prego_1e-3": "#aec7e8",  # light blue
    "full_trial_pl__pos10_prego_5e-2": "#9467bd",  # purple
    "full_trial_pl__pos10_prego_1":    "#8c564b",  # brown
}

N_REPLICATES = 5
N_WARMUP_BATCHES = 12000
EXPERIMENT = "3702f54"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _make_args_namespace(label: str) -> argparse.Namespace:
    """Build argparse.Namespace with the correct per-cell CLI flags."""
    defaults = dict(
        n_warmup_batches=N_WARMUP_BATCHES,
        n_adversary_batches=0,
        batch_size=250,
        n_replicates=N_REPLICATES,
        nn_output_jerk=0.0,
        seed=42,
        hidden_type="gru",
        sisu_gating="additive",
        loss_update_enabled=False,
        loss_update_ratio=0.5,
        # Shared loss weights (per RUN_PLAN.md)
        effector_pos_running=1.0,
        effector_hold_pos=1.0,
        effector_hold_vel=0.0,
        effector_pos_late_weight=0.0,
        effector_vel_late=0.0,
        effector_final_vel=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        p_catch_trial=0.5,
        nn_output=1e-5,
        nn_hidden=1e-5,
        nn_hidden_derivative=1e-3,  # explicit on 3702f54 (and f47abb1 baselines)
        nn_output_pre_go=0.0,
        nn_hidden_derivative_pre_go=0.0,
        # Power-law schedule (per-cell overrides)
        effector_pos_running_schedule="flat",
        effector_hold_pos_schedule="flat",
        position_powerlaw_power=6.0,
        controller_lr=1e-4,
    )
    defaults.update(CELL_EXTRA_ARGS[label])
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cell_model(label: str, artifact_base: Path):
    """Load the warmup model for a cell. `warmup_model.eqx` is the canonical final
    model since n_adversary_batches=0 across this matrix (Bug: a517040).
    """
    experiment = CELL_ARTIFACT_EXPERIMENT[label]
    cell_dir = artifact_base / experiment / "runs" / label
    eqx_path = cell_dir / "warmup_model.eqx"
    if not eqx_path.exists():
        raise FileNotFoundError(f"warmup_model.eqx not found: {eqx_path}")

    args = _make_args_namespace(label)
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(42))
    task = pair.task

    model, _ = load_with_hyperparameters(
        eqx_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    n_reps = _count_replicates(model)
    return model, task, n_reps, eqx_path


def _count_replicates(model) -> int:
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim >= 3:
            return int(leaf.shape[0])
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim == 2:
            return 1
    return 1


def load_warmup_history(label: str, artifact_base: Path) -> TaskTrainerHistory:
    """Load `warmup_history.eqx` for per-cell loss-decomposition figure."""
    experiment = CELL_ARTIFACT_EXPERIMENT[label]
    cell_dir = artifact_base / experiment / "runs" / label
    history_path = cell_dir / "warmup_history.eqx"
    if not history_path.exists():
        raise FileNotFoundError(f"warmup_history.eqx not found: {history_path}")

    args = _make_args_namespace(label)
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(42))

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_zero_pert_trials(task, *, sisu: float = 0.5):
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
# Kinematics
# ---------------------------------------------------------------------------

# Pre-go window length in steps (200 ms at dt=10 ms).
PRE_GO_WINDOW_STEPS = 20


def compute_kinematics_per_replicate(states, trial_specs) -> dict[str, np.ndarray]:
    """Compute per-replicate kinematic metrics.

    Returns dict with arrays for:
      peak_forward_velocity, time_to_peak (after go, in steps after go cue),
      forward_vel_profile, pos_forward_profile, hold_drift_mm (max over full hold),
      pre_go_drift_mm (RMS over last PRE_GO_WINDOW_STEPS pre-go steps),
      pre_go_drift_mean_mm (mean forward pos over pre-go window),
      go_idx.
    """
    pos = states.mechanics.effector.pos  # (n_rep, n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel

    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value
    goal = goal_seq[:, -1, :]

    go_idx = trial_specs.timeline.epoch_bounds[:, 2]

    n_rep, n_trials, n_steps, _ = pos.shape
    t_arr = jnp.arange(n_steps)
    after_go = t_arr[None, None, :] >= go_idx[None, :, None]
    before_go = t_arr[None, None, :] < go_idx[None, :, None]
    pre_go_window = (
        (t_arr[None, None, :] < go_idx[None, :, None])
        & (t_arr[None, None, :] >= (go_idx[None, :, None] - PRE_GO_WINDOW_STEPS))
    )

    def _pos_at_go(pos_rep, go_arr):
        return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_arr)

    pos_at_go = jax.vmap(_pos_at_go, in_axes=(0, None))(pos, go_idx)
    direction = goal[None, :, :] - pos_at_go
    d_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    d_unit = direction / jnp.maximum(d_norm, 1e-12)

    v_fwd = jnp.sum(vel * d_unit[:, :, None, :], axis=-1)
    v_fwd_post_go = jnp.where(after_go, v_fwd, 0.0)

    peak_fwd = jnp.max(v_fwd_post_go, axis=-1)

    # Time-to-peak measured from go cue (steps after go).
    abs_argmax = jnp.argmax(v_fwd_post_go, axis=-1)  # (n_rep, n_trials)
    time_to_peak_after_go = abs_argmax - go_idx[None, :]  # (n_rep, n_trials)
    time_to_peak_after_go = jnp.maximum(time_to_peak_after_go, 0)

    pos_at_start = pos[:, :, 0, :]
    pos_rel = pos - pos_at_start[:, :, None, :]
    pos_fwd = jnp.sum(pos_rel * d_unit[:, :, None, :], axis=-1)

    # Whole-hold (any time before go) max forward displacement (anticipation peak).
    pos_fwd_pre_go = jnp.where(before_go, pos_fwd, -jnp.inf)
    hold_drift_m = jnp.max(pos_fwd_pre_go, axis=-1)
    hold_drift_m = jnp.where(jnp.isinf(hold_drift_m), 0.0, hold_drift_m)
    hold_drift_mm = hold_drift_m * 1000.0

    # Pre-go window (-200 ms to 0): RMS forward position (positive = anticipation).
    pos_fwd_window_masked = jnp.where(pre_go_window, pos_fwd, 0.0)
    window_counts = jnp.sum(pre_go_window, axis=-1)  # (1, n_trials, ) ->; rely on broadcasting
    window_counts = jnp.maximum(window_counts, 1)
    pre_go_rms_m = jnp.sqrt(
        jnp.sum(pos_fwd_window_masked ** 2, axis=-1) / window_counts
    )
    pre_go_mean_m = jnp.sum(pos_fwd_window_masked, axis=-1) / window_counts

    pre_go_rms_mm = pre_go_rms_m * 1000.0
    pre_go_mean_mm = pre_go_mean_m * 1000.0

    return {
        "peak_forward_velocity": np.array(peak_fwd),
        "time_to_peak_after_go": np.array(time_to_peak_after_go),
        "forward_vel_profile": np.array(v_fwd),
        "pos_forward_profile": np.array(pos_fwd),
        "hold_drift_mm": np.array(hold_drift_mm),
        "pre_go_rms_mm": np.array(pre_go_rms_mm),
        "pre_go_mean_mm": np.array(pre_go_mean_mm),
        "go_idx": np.array(go_idx),
    }


# ---------------------------------------------------------------------------
# Per-cell aggregate statistics
# ---------------------------------------------------------------------------

def _within_cell_mean_pairwise_rmse(profiles: np.ndarray) -> float:
    """Mean pairwise RMSE between all distinct pairs of replicates."""
    n_rep = profiles.shape[0]
    rmse_vals = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            diff = profiles[i] - profiles[j]
            rmse_vals.append(float(np.sqrt(np.mean(diff ** 2))))
    return float(np.mean(rmse_vals)) if rmse_vals else float("nan")


def compute_cell_stats(km: dict[str, np.ndarray]) -> dict:
    """Aggregate kinematics to per-replicate scalars + per-cell stats."""
    peak_vel_per_rep = km["peak_forward_velocity"].mean(axis=-1)
    hold_drift_per_rep = km["hold_drift_mm"].mean(axis=-1)
    pre_go_rms_per_rep = km["pre_go_rms_mm"].mean(axis=-1)
    pre_go_mean_per_rep = km["pre_go_mean_mm"].mean(axis=-1)
    ttp_per_rep = km["time_to_peak_after_go"].mean(axis=-1)

    # Within-cell pairwise velocity-RMSE (absolute m/s, primary metric per issue request).
    # Bug: 06f7faf — align per-trial profiles to each trial's go cue BEFORE
    # averaging across trials, and trim to the full-support window so the
    # NaN-edge columns do not contaminate the RMSE.
    aligned_vel, _center = align_trials(km["forward_vel_profile"], km["go_idx"])
    vel_profiles, _sl = replicate_mean_curves(aligned_vel)  # (n_rep, n_kept_steps)
    within_rmse_vel = _within_cell_mean_pairwise_rmse(vel_profiles)

    mean_pv = float(peak_vel_per_rep.mean())
    sd_pv = float(peak_vel_per_rep.std(ddof=1)) if len(peak_vel_per_rep) > 1 else 0.0

    return {
        "peak_vel_per_rep": peak_vel_per_rep.tolist(),
        "hold_drift_per_rep": hold_drift_per_rep.tolist(),
        "pre_go_rms_per_rep": pre_go_rms_per_rep.tolist(),
        "pre_go_mean_per_rep": pre_go_mean_per_rep.tolist(),
        "time_to_peak_per_rep": ttp_per_rep.tolist(),
        "mean_peak_velocity": mean_pv,
        "sd_peak_velocity": sd_pv,
        "mean_hold_drift_mm": float(hold_drift_per_rep.mean()),
        "sd_hold_drift_mm": float(hold_drift_per_rep.std(ddof=1)) if len(hold_drift_per_rep) > 1 else 0.0,
        "mean_pre_go_rms_mm": float(pre_go_rms_per_rep.mean()),
        "sd_pre_go_rms_mm": float(pre_go_rms_per_rep.std(ddof=1)) if len(pre_go_rms_per_rep) > 1 else 0.0,
        "mean_pre_go_drift_mm": float(pre_go_mean_per_rep.mean()),
        "sd_pre_go_drift_mm": float(pre_go_mean_per_rep.std(ddof=1)) if len(pre_go_mean_per_rep) > 1 else 0.0,
        "mean_time_to_peak_steps": float(ttp_per_rep.mean()),
        "sd_time_to_peak_steps": float(ttp_per_rep.std(ddof=1)) if len(ttp_per_rep) > 1 else 0.0,
        "within_cell_vel_rmse": within_rmse_vel,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_forward_velocity_profile_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Forward velocity time series faceted by cell, mean +/- SD across replicates."""
    labels_present = [l for l in CELL_LABELS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    # Bug: 06f7faf — profile_comparison_grid applies shared y-axes by default
    # so cells are visually comparable across panels.
    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        vertical_spacing=0.025,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        v_fwd = km["forward_vel_profile"]  # (n_rep, n_trials, n_steps)
        go_idx = km["go_idx"]
        n_rep, n_trials, n_steps = v_fwd.shape
        color = CELL_COLORS[label]

        # Bug: 06f7faf — go-cue alignment + pooled (replicate, trial) band.
        # Each trial is re-locked to its own go cue (column = center); the
        # plotted band represents inter-trial variability over the full pooled
        # (n_rep * n_trials) population. Aggregator trims the choppy NaN-edge
        # columns to the strict full-support window and returns the slice.
        aligned_v, center = align_trials(v_fwd, go_idx)
        mean, lower, upper, sl = pooled_trial_mean_with_band(aligned_v, band="sd")
        # Time axis relative to go cue (t=0 at go), trimmed to match the curve.
        t = ((np.arange(aligned_v.shape[-1]) - center) * dt)[sl]

        # Upper band
        fig.add_trace(go.Scatter(
            x=t, y=upper, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        # Lower band w/ fill
        fig.add_trace(go.Scatter(
            x=t, y=lower, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor=_color_rgba(color, 0.25),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        # Mean line
        fig.add_trace(go.Scatter(
            x=t, y=mean, mode="lines",
            line=dict(color=color, width=2),
            name=CELL_DISPLAY_NAMES[label], showlegend=False,
        ), row=row, col=1)

        # Go cue is now at t=0 by construction (per-trial alignment).
        fig.add_vline(
            x=0.0,
            line=dict(color="black", dash="dash", width=1),
            row=row, col=1,
        )

    fig.update_layout(
        title=(
            "Forward velocity profiles (go-cue-aligned, pooled trial mean ± SD) — pre-go matrix (3702f54)<br>"
            "<sup>Each trial re-locked to its go cue (t=0). Band is across pooled "
            "(replicate × trial) samples. Baselines at top. Bug: 06f7faf.</sup>"
        ),
        width=1000,
        height=180 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd vel (m/s)", row=row, col=1)

    return fig


def make_hold_drift_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Pre-go forward position (anticipation) per cell."""
    labels_present = [l for l in CELL_LABELS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    # Bug: 06f7faf — shared y-axes via profile_comparison_grid; hold-drift
    # plots compare anticipation amplitude across cells, so visual comparability
    # is essential.
    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        vertical_spacing=0.025,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        pos_fwd = km["pos_forward_profile"]  # (n_rep, n_trials, n_steps)
        go_idx = km["go_idx"]
        n_rep, n_trials, n_steps = pos_fwd.shape
        color = CELL_COLORS[label]

        # Bug: 06f7faf — align to go cue, then clip to the pre-go window
        # (t in [-window, 0]). Pooled-trial band; aggregator trims the
        # choppy NaN-edge columns to full-support before reducing.
        aligned_pos, center = align_trials(pos_fwd, go_idx)
        mean_m, lower_m, upper_m, sl = pooled_trial_mean_with_band(aligned_pos, band="sd")
        # Convert to mm
        mean = mean_m * 1000.0
        lower = lower_m * 1000.0
        upper = upper_m * 1000.0

        # Time axis relative to go cue, trimmed to the full-support window.
        t_rel = ((np.arange(aligned_pos.shape[-1]) - center) * dt)[sl]
        # Clip to pre-go window (-PRE_GO_WINDOW_STEPS to 0)
        keep = (t_rel >= -PRE_GO_WINDOW_STEPS * dt) & (t_rel <= 0.0)
        t_pre = t_rel[keep]
        mean = mean[keep]
        lower = lower[keep]
        upper = upper[keep]

        fig.add_trace(go.Scatter(
            x=t_pre, y=upper, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t_pre, y=lower, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor=_color_rgba(color, 0.25),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t_pre, y=mean, mode="lines",
            line=dict(color=color, width=2),
            name=CELL_DISPLAY_NAMES[label], showlegend=False,
        ), row=row, col=1)

        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1), row=row, col=1)
        # Mark pre-go window onset (-PRE_GO_WINDOW_STEPS*dt)
        fig.add_vline(
            x=-PRE_GO_WINDOW_STEPS * dt,
            line=dict(color="red", dash="dot", width=1),
            row=row, col=1,
        )

    fig.update_layout(
        title=(
            "Pre-go forward position drift (anticipation, go-cue-aligned) — pre-go matrix (3702f54)<br>"
            "<sup>Pooled (replicate × trial) mean ± SD. Red dotted = pre-go window onset (-200 ms). "
            "t=0 is the go cue per trial. Baselines at top. Bug: 06f7faf.</sup>"
        ),
        width=1000,
        height=180 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd pos (mm)", row=row, col=1)

    return fig


def make_peak_velocity_figure(cell_stats: dict[str, dict]) -> go.Figure:
    """Strip/violin plot of per-replicate peak forward velocity."""
    fig = go.Figure()
    for label in CELL_LABELS:
        if label not in cell_stats:
            continue
        stats = cell_stats[label]
        pvs = stats["peak_vel_per_rep"]
        color = CELL_COLORS[label]
        fig.add_trace(go.Violin(
            y=pvs,
            name=CELL_DISPLAY_NAMES[label],
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.3,
            pointpos=-1.2,
            line_color=color,
            fillcolor=_color_rgba(color, 0.35),
            marker=dict(color=color, size=8),
            showlegend=False,
        ))

    fig.update_layout(
        title=(
            "Peak forward velocity per replicate — pre-go matrix (3702f54)<br>"
            "<sup>Baselines at left. f47abb1 sanity range: 0.7-1.0 m/s</sup>"
        ),
        yaxis_title="Peak forward velocity (m/s)",
        xaxis_title="Cell",
        width=1300,
        height=550,
        margin=dict(l=70, r=40, t=80, b=200),
    )
    # Reference lines (sanity range)
    fig.add_hline(y=0.7, line=dict(color="grey", dash="dot", width=1), annotation_text="0.7 m/s", annotation_position="left")
    fig.add_hline(y=1.0, line=dict(color="grey", dash="dot", width=1), annotation_text="1.0 m/s", annotation_position="left")

    return fig


def make_summary_metrics_figure(cell_stats: dict[str, dict]) -> go.Figure:
    """Bar plot of key per-cell summary metrics: vel-RMSE, hold drift, pre-go RMS, peak vel, TTP."""
    labels_present = [l for l in CELL_LABELS if l in cell_stats]
    display_names = [CELL_DISPLAY_NAMES[l] for l in labels_present]
    colors = [CELL_COLORS[l] for l in labels_present]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Within-cell pairwise velocity-RMSE (m/s)",
            "Hold-period peak drift (mm) - whole hold",
            "Pre-go forward drift RMS over [-200ms, 0] (mm)",
            "Peak forward velocity (m/s, mean +/- SD)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    vel_rmse = [cell_stats[l]["within_cell_vel_rmse"] for l in labels_present]
    hold_drift = [cell_stats[l]["mean_hold_drift_mm"] for l in labels_present]
    hold_drift_sd = [cell_stats[l]["sd_hold_drift_mm"] for l in labels_present]
    pre_go_rms = [cell_stats[l]["mean_pre_go_rms_mm"] for l in labels_present]
    pre_go_rms_sd = [cell_stats[l]["sd_pre_go_rms_mm"] for l in labels_present]
    peak_vel = [cell_stats[l]["mean_peak_velocity"] for l in labels_present]
    peak_vel_sd = [cell_stats[l]["sd_peak_velocity"] for l in labels_present]

    fig.add_trace(go.Bar(
        x=display_names, y=vel_rmse, marker_color=colors,
        text=[f"{v:.4f}" for v in vel_rmse], textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=display_names, y=hold_drift,
        error_y=dict(type="data", array=hold_drift_sd),
        marker_color=colors,
        text=[f"{v:.2f}" for v in hold_drift], textposition="outside",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=display_names, y=pre_go_rms,
        error_y=dict(type="data", array=pre_go_rms_sd),
        marker_color=colors,
        text=[f"{v:.2f}" for v in pre_go_rms], textposition="outside",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=display_names, y=peak_vel,
        error_y=dict(type="data", array=peak_vel_sd),
        marker_color=colors,
        text=[f"{v:.3f}" for v in peak_vel], textposition="outside",
    ), row=2, col=2)

    # Threshold line at 0.5 mm on the pre-go RMS panel
    fig.add_hline(y=0.5, line=dict(color="red", dash="dash", width=1.5), row=2, col=1,
                  annotation_text="0.5 mm target", annotation_position="top right")

    fig.update_layout(
        title=(
            "Summary metrics per cell — pre-go matrix (3702f54)<br>"
            "<sup>Baselines (f47abb1) anchored at left. All quantities absolute (not ratios).</sup>"
        ),
        width=1600,
        height=900,
        showlegend=False,
        margin=dict(l=70, r=40, t=100, b=220),
    )
    fig.update_yaxes(title_text="Vel-RMSE (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Hold drift (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Pre-go RMS (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Peak vel (m/s)", row=2, col=2)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(tickangle=-45, row=r, col=c)

    return fig


def make_training_loss_per_term_figure(histories: dict[str, TaskTrainerHistory]) -> go.Figure:
    """Per-term training loss decomposition for all loaded cells."""
    all_term_keys: set[str] = set()
    term_data: dict[str, dict[str, np.ndarray]] = {}
    for label, history in histories.items():
        flat = history.loss.flatten(apply_weights=True)
        term_data[label] = {k: np.array(v) for k, v in flat.items()}
        all_term_keys.update(term_data[label].keys())

    term_keys = sorted(all_term_keys)
    n_terms = len(term_keys)
    if n_terms == 0:
        return go.Figure()

    n_cols = min(4, n_terms)
    n_rows = (n_terms + n_cols - 1) // n_cols

    fig = make_subplots(
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
        for label in CELL_LABELS:
            if label not in term_data or term_key not in term_data[label]:
                continue
            term_vals = term_data[label][term_key]  # (n_batches, n_replicates)
            n_batches = term_vals.shape[0]
            x = np.arange(n_batches)
            mean = term_vals.mean(axis=1)

            # Replace zeros with tiny floor so log-log is meaningful.
            mean = np.where(mean > 0, mean, 1e-30)

            color = CELL_COLORS[label]
            display_name = CELL_DISPLAY_NAMES[label]
            show_legend = (term_idx == 0)

            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines",
                name=display_name,
                line=dict(color=color, width=1.5),
                legendgroup=label,
                showlegend=show_legend,
            ), row=row, col=col)

    fig.update_layout(
        title=(
            "Per-term weighted training loss vs batch — pre-go matrix (3702f54)<br>"
            "<sup>Baselines (f47abb1) included. nn_output_pre_go term present only for cells with prego > 0. "
            "Cross-schedule absolutes not comparable.</sup>"
        ),
        width=320 * n_cols + 240,
        height=290 * n_rows + 120,
        margin=dict(l=60, r=240, t=100, b=60),
    )
    for i in range(1, n_terms + 1):
        r = (i - 1) // n_cols + 1
        c = (i - 1) % n_cols + 1
        fig.update_yaxes(type="log", row=r, col=c)
        fig.update_xaxes(type="log", row=r, col=c)

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
        help="Base directory above experiment dirs (default: <repo>/_artifacts)",
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
    parser.add_argument(
        "--skip-training-loss",
        action="store_true",
        help="Skip the per-term training-loss figure (slow to load histories).",
    )
    args = parser.parse_args()

    artifact_base = args.artifact_base or (REPO_ROOT / "_artifacts")
    results_base = REPO_ROOT / "results" / EXPERIMENT

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")

    notes_dir = results_base / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    cell_stats: dict[str, dict] = {}
    cell_kms: dict[str, dict] = {}
    input_artifacts: list[dict] = []

    for label in CELL_LABELS:
        print(f"\n[{label}] Loading model ...", flush=True)
        try:
            model, task, n_reps, eqx_path = load_cell_model(label, artifact_base)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        except Exception as e:
            import traceback
            print(f"  FAILED loading: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        print(f"  Loaded. n_replicates={n_reps}")
        input_artifacts.append({"path": str(eqx_path), "role": f"warmup_model:{label}"})

        try:
            trial_specs = build_zero_pert_trials(task, sisu=args.sisu)
            n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
            print(f"  Evaluating {n_reps} replicates on {n_trials} trials ...", flush=True)
            states = eval_ensemble(
                task, model, trial_specs,
                key=jr.PRNGKey(args.eval_seed), n_replicates=n_reps,
            )
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
            f"  peak_vel: {stats['mean_peak_velocity']:.4f} +/- {stats['sd_peak_velocity']:.4f} m/s  "
            f"vel-RMSE: {stats['within_cell_vel_rmse']:.4f} m/s  "
            f"hold_drift: {stats['mean_hold_drift_mm']:.2f} mm  "
            f"pre_go_RMS: {stats['mean_pre_go_rms_mm']:.2f} mm  "
            f"TTP: {stats['mean_time_to_peak_steps']:.1f} steps"
        )

    if not cell_stats:
        print("\nNo cells loaded -- aborting.")
        return

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n--- Building figures ---")

    # Figure 1: Forward velocity profiles
    fig_fv = make_forward_velocity_profile_figure(cell_kms)
    spec_fv = {
        "figure_kind": "forward_velocity_profile_time_series_go_aligned",
        "experiment": EXPERIMENT,
        "inputs": input_artifacts,
        "transform": [
            {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
            {"name": "forward_velocity_projection_onto_reach_axis", "kwargs": {}},
            {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
            {"name": "trim_to_full_support", "kwargs": {"min_coverage": 1.0}},
            {"name": "pooled_trial_nanmean_with_sd_band", "kwargs": {}},
        ],
        "plot_kwargs": {
            "cells": CELL_LABELS,
            "n_replicates": N_REPLICATES,
            "sisu": args.sisu,
            "pert_scale": 0.0,
            "dt": 0.01,
            "ordering_note": "Baselines (lit__post_nojerk, lit__full_nojerk) at top",
            "alignment": "go_cue_per_trial",
            "band_semantic": "pooled_replicate_trial_sd",
            "shared_yaxes": "all",
        },
        "fix_note": "Bug: 06f7faf — go-cue alignment fix + trim-to-full-support + shared y-axes across cells.",
    }
    fv_out = save_figure(
        fig=fig_fv, spec=spec_fv,
        package="rlrmp", experiment=EXPERIMENT, topic="forward_velocity_profiles",
        extra_packages=["rlrmp"],
    )
    print(f"  forward_velocity_profiles spec: {fv_out['spec_path']}")
    print(f"  forward_velocity_profiles HTML: {fv_out['render_path']}")

    # Figure 2: Hold drift profiles
    fig_hd = make_hold_drift_figure(cell_kms)
    spec_hd = {
        "figure_kind": "hold_drift_profile_pre_go_position_go_aligned",
        "experiment": EXPERIMENT,
        "inputs": input_artifacts,
        "transform": [
            {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
            {"name": "forward_position_projection_onto_reach_axis", "kwargs": {}},
            {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
            {"name": "trim_to_full_support", "kwargs": {"min_coverage": 1.0}},
            {"name": "pooled_trial_nanmean_with_sd_band", "kwargs": {}},
            {"name": "clip_to_pre_go_window", "kwargs": {"window_steps": PRE_GO_WINDOW_STEPS}},
        ],
        "plot_kwargs": {
            "cells": CELL_LABELS,
            "n_replicates": N_REPLICATES,
            "sisu": args.sisu,
            "pert_scale": 0.0,
            "dt": 0.01,
            "pre_go_window_steps": PRE_GO_WINDOW_STEPS,
            "ordering_note": "Baselines (lit__post_nojerk, lit__full_nojerk) at top",
            "alignment": "go_cue_per_trial",
            "band_semantic": "pooled_replicate_trial_sd",
            "shared_yaxes": "all",
        },
        "fix_note": "Bug: 06f7faf — go-cue alignment fix + trim-to-full-support + shared y-axes across cells.",
    }
    hd_out = save_figure(
        fig=fig_hd, spec=spec_hd,
        package="rlrmp", experiment=EXPERIMENT, topic="hold_drift_profiles",
        extra_packages=["rlrmp"],
    )
    print(f"  hold_drift_profiles spec: {hd_out['spec_path']}")
    print(f"  hold_drift_profiles HTML: {hd_out['render_path']}")

    # Figure 3: Peak velocity distributions
    fig_pv = make_peak_velocity_figure(cell_stats)
    spec_pv = {
        "figure_kind": "peak_velocity_distributions_violin",
        "experiment": EXPERIMENT,
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
            "ordering_note": "Baselines (f47abb1) at left",
        },
        "cell_stats": {
            label: {
                "mean_peak_velocity": stats["mean_peak_velocity"],
                "sd_peak_velocity": stats["sd_peak_velocity"],
                "peak_vel_per_rep": stats["peak_vel_per_rep"],
            }
            for label, stats in cell_stats.items()
        },
    }
    pv_out = save_figure(
        fig=fig_pv, spec=spec_pv,
        package="rlrmp", experiment=EXPERIMENT, topic="peak_velocity_distributions",
        extra_packages=["rlrmp"],
    )
    print(f"  peak_velocity_distributions spec: {pv_out['spec_path']}")
    print(f"  peak_velocity_distributions HTML: {pv_out['render_path']}")

    # Figure 4: Summary metrics
    fig_sm = make_summary_metrics_figure(cell_stats)
    spec_sm = {
        "figure_kind": "summary_metrics_bar_panel",
        "experiment": EXPERIMENT,
        "inputs": input_artifacts,
        "transform": [
            {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
            {"name": "compute_kinematics_per_replicate", "kwargs": {}},
            {"name": "aggregate_per_cell_stats", "kwargs": {}},
        ],
        "plot_kwargs": {
            "cells": CELL_LABELS,
            "n_replicates": N_REPLICATES,
            "sisu": args.sisu,
            "pert_scale": 0.0,
            "pre_go_window_steps": PRE_GO_WINDOW_STEPS,
            "metrics_shown": [
                "within_cell_vel_rmse",
                "mean_hold_drift_mm",
                "mean_pre_go_rms_mm",
                "mean_peak_velocity",
            ],
            "ordering_note": "Baselines (f47abb1) at left",
        },
        "cell_stats": {
            label: {
                k: stats[k] for k in (
                    "within_cell_vel_rmse",
                    "mean_hold_drift_mm",
                    "sd_hold_drift_mm",
                    "mean_pre_go_rms_mm",
                    "sd_pre_go_rms_mm",
                    "mean_pre_go_drift_mm",
                    "sd_pre_go_drift_mm",
                    "mean_peak_velocity",
                    "sd_peak_velocity",
                    "mean_time_to_peak_steps",
                    "sd_time_to_peak_steps",
                )
            }
            for label, stats in cell_stats.items()
        },
    }
    sm_out = save_figure(
        fig=fig_sm, spec=spec_sm,
        package="rlrmp", experiment=EXPERIMENT, topic="summary_metrics",
        extra_packages=["rlrmp"],
    )
    print(f"  summary_metrics spec: {sm_out['spec_path']}")
    print(f"  summary_metrics HTML: {sm_out['render_path']}")

    # Figure 5: Training loss per term
    if not args.skip_training_loss:
        print("\n--- Loading warmup histories for per-term loss figure ---")
        histories: dict[str, TaskTrainerHistory] = {}
        history_inputs: list[dict] = []
        for label in CELL_LABELS:
            if label not in cell_stats:
                continue
            try:
                history = load_warmup_history(label, artifact_base)
                histories[label] = history
                history_path = (
                    artifact_base / CELL_ARTIFACT_EXPERIMENT[label] / "runs" / label / "warmup_history.eqx"
                )
                history_inputs.append({"path": str(history_path), "role": f"warmup_history:{label}"})
                print(f"  loaded history: {label}")
            except Exception as e:
                print(f"  FAILED loading history for {label}: {type(e).__name__}: {e}")
                continue

        if histories:
            fig_tl = make_training_loss_per_term_figure(histories)
            spec_tl = {
                "figure_kind": "training_loss_per_term_multiline",
                "experiment": EXPERIMENT,
                "inputs": history_inputs,
                "transform": [
                    {"name": "load_warmup_history_fbx", "kwargs": {"header_lines": 1}},
                    {"name": "TermTree.flatten", "kwargs": {"apply_weights": True}},
                    {"name": "mean_across_replicates", "kwargs": {"axis": 1}},
                ],
                "plot_kwargs": {
                    "cells": CELL_LABELS,
                    "n_replicates": N_REPLICATES,
                    "n_warmup_batches": N_WARMUP_BATCHES,
                    "note": (
                        "nn_output_pre_go term present only for cells with prego > 0 "
                        "(cells 3-8 of 3702f54). Baselines and prego=0 cells lack that term."
                    ),
                },
            }
            tl_out = save_figure(
                fig=fig_tl, spec=spec_tl,
                package="rlrmp", experiment=EXPERIMENT, topic="training_loss_per_term",
                extra_packages=["rlrmp"],
            )
            print(f"  training_loss_per_term spec: {tl_out['spec_path']}")
            print(f"  training_loss_per_term HTML: {tl_out['render_path']}")

    # -----------------------------------------------------------------------
    # Summary table + JSON
    # -----------------------------------------------------------------------
    print("\n=== SUMMARY (3702f54 pre-go motor mask matrix) ===\n")
    header = (
        f"{'Cell':36s} {'Vel-RMSE':>10} {'HoldDrift':>10} {'PreGoRMS':>10} "
        f"{'PeakVel':>10} {'TTP':>6}"
    )
    print(header)
    print("-" * len(header))
    for label in CELL_LABELS:
        if label not in cell_stats:
            print(f"  {CELL_DISPLAY_NAMES[label]:36s} SKIPPED")
            continue
        s = cell_stats[label]
        print(
            f"  {CELL_DISPLAY_NAMES[label][:34]:34s}  "
            f"{s['within_cell_vel_rmse']:>9.4f}  "
            f"{s['mean_hold_drift_mm']:>9.2f}  "
            f"{s['mean_pre_go_rms_mm']:>9.2f}  "
            f"{s['mean_peak_velocity']:>9.4f}  "
            f"{s['mean_time_to_peak_steps']:>5.1f}"
        )

    # Save per-cell stats as JSON
    stats_json_path = notes_dir / "analysis_data.json"
    json_data = {
        "experiment": EXPERIMENT,
        "sisu": args.sisu,
        "pre_go_window_steps": PRE_GO_WINDOW_STEPS,
        "baselines": BASELINE_CELLS,
        "new_cells": NEW_CELLS,
        "metric_descriptions": {
            "within_cell_vel_rmse": "Mean pairwise forward-velocity RMSE across replicate pairs within the cell (m/s, absolute).",
            "mean_hold_drift_mm": "Max forward effector displacement during the whole hold period (mm), averaged over replicates.",
            "mean_pre_go_rms_mm": "Root-mean-square forward effector position over the last 200 ms before go (mm).",
            "mean_pre_go_drift_mm": "Mean forward effector position over the last 200 ms before go (mm; signed).",
            "mean_peak_velocity": "Peak forward velocity after go cue (m/s).",
            "mean_time_to_peak_steps": "Steps from go cue to peak velocity (dt=10 ms).",
        },
        "cells": cell_stats,
    }
    with open(stats_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved stats JSON: {stats_json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
