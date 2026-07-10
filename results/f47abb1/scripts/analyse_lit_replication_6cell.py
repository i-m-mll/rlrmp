# TODO: relocate to results/f47abb1/scripts/ — per CLAUDE.md script-placement convention
"""Variance + anticipation analysis for the lit-replication 6-cell matrix (f47abb1).

Bug: 06f7faf — go-cue alignment fix. Velocity-RMSE-ratio and position-RMSE-ratio
(primary + secondary metrics) and the velocity/hold-drift profile figures now
use per-trial go-cue alignment via `rlrmp.analysis.math.trial_alignment`.


Tests whether faithful Chaisanguanthum & Shenoy 2019 (C&S) loss schedule gives
better velocity-RMSE ratios than the production loss.

Two crossed design axes:
  - Jerk regulariser: on (nn_output_jerk=1e5) vs off (0)
  - Position schedule: flat / post-go (t/T)^6 / full-trial (t/T)^6

Computes per-(cell x replicate):
  - Peak forward velocity (m/s)
  - Hold-period / pre-go drift (mm): forward displacement during pre-go window
  - Forward velocity profile over time (all 8 validation reach directions)
  - Position profile over time (all 8 validation reach directions)

Then per cell:
  - Mean +/- SD peak forward velocity across replicates
  - CV (SD/mean of peak vel across replicates) -- auxiliary metric only
  - Within-cell velocity-profile RMSE ratio: within / nearest-across-cell -- PRIMARY metric
  - Within-cell position-profile RMSE ratio: same on position profiles -- secondary
  - Mean hold drift

The **primary** variance metric is the pairwise profile-RMSE ratio
(within-cell / nearest-across-cell), matching the prior
``baseline_jerk_vrnn_matrix`` metric (GRU/jerk prior best: 0.758).
CV (SD/mean of peak velocity scalar) is an auxiliary summary only.

Decision criterion: velocity-RMSE ratio < 0.50 (prior best = 0.758 for GRU/jerk baseline).

**Cross-schedule absolute loss note**: absolute loss values are NOT comparable across
position schedule shapes. The powerlaw (t/T)^6 concentrates ~98% of position weight in
the last 30% of the trial, so the weighted sum is structurally lower than for flat.
Compare WITHIN schedule shape (jerk vs no-jerk at flat / post / full separately).

Produces four HTML figures:
  1. peak_velocity_distributions  -- violin / box with all 6 cells (one replicate = one point)
  2. forward_velocity_profiles    -- time-series per cell, one trace per replicate
  3. hold_drift_profiles          -- pre-go position (forward direction) per cell
  4. rmse_ratio_comparison        -- bar chart of velocity-RMSE and position-RMSE ratios

Usage (from repo root):
    /path/to/.venv/bin/python scripts/analyse_lit_replication_6cell.py
"""

from __future__ import annotations
from rlrmp.analysis.multi_cell_driver import compute_kinematics_per_replicate
from rlrmp.viz.colors import hex_to_rgba as _color_rgba

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
from jax_cookbook import load_with_hyperparameters
from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476 -- project-config routing
from plotly.subplots import make_subplots

from rlrmp.analysis.math.trial_alignment import (
    align_trials,
    pooled_trial_mean_with_band,
    replicate_mean_curves,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT  # Bug: 8404108 — was __file__-relative
from rlrmp.train.minimax import build_hps
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz import profile_comparison_grid

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
# Shared defaults: hidden_type=gru, n_warmup_batches=12000, n_adversary_batches=0,
#   batch_size=250, n_replicates=5, seed=42, effector_hold_pos=1.0,
#   effector_hold_vel=0.0, effector_pos_running=1.0, effector_pos_late_weight=0.0,
#   effector_vel_late=0.0, effector_final_vel=0.0, p_catch_trial=0.5,
#   nn_output=1e-5, nn_hidden=1e-5, nn_output_pre_go=0.0, nn_hidden_derivative_pre_go=0.0,
#   loss_update_enabled=False, position_powerlaw_power=6.0.
CELL_EXTRA_ARGS: dict[str, dict] = {
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
EXPERIMENT = "f47abb1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _make_args_namespace(label: str) -> argparse.Namespace:
    """Build argparse.Namespace with the correct per-cell CLI flags."""
    defaults = dict(
        n_warmup_batches=N_WARMUP_BATCHES,
        n_adversary_batches=0,
        batch_size=250,
        n_replicates=N_REPLICATES,
        nn_output_jerk=0.0,  # per-cell override from CELL_EXTRA_ARGS
        seed=42,
        hidden_type="gru",
        sisu_gating="additive",
        loss_update_enabled=False,
        loss_update_ratio=0.5,
        # f47abb1 shared loss weights
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
        # Bug: f47abb1 — actual training-time weight (inspected from saved
        # warmup_history.eqx). run.json under-reports this CLI flag.
        # Affects only loss_func structure (not model weights), but kept here
        # for consistency with plot_training_loss_lit_replication.py.
        nn_hidden_derivative=0.001,
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
    """Load the trained model for a lit-replication 6-cell label.

    For the f47abb1 matrix `n_adversary_batches=0`, so the adversarial phase
    did not actually run and `adversarial_model.eqx` is the warmup model
    re-wrapped with adversary state (which doesn't match the local skeleton
    after the warmup_model save). Prefer `warmup_model.eqx`, which loads
    cleanly against the current skeleton. Bug: f47abb1.

    Returns (model, task, n_replicates).
    """
    cell_dir = artifact_base / label
    # Prefer warmup_model.eqx (clean PyTree). Fall back to adversarial_model.eqx
    # only if warmup is missing (would never be the case for this matrix).
    warmup_path = cell_dir / "warmup_model.eqx"
    adv_path = cell_dir / "adversarial_model.eqx"
    if warmup_path.exists():
        eqx_path = warmup_path
    elif adv_path.exists():
        eqx_path = adv_path
    else:
        raise FileNotFoundError(
            f"Neither warmup_model.eqx nor adversarial_model.eqx in {cell_dir}"
        )

    args = _make_args_namespace(label)
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(42))
    task = pair.task

    model, _ = load_with_hyperparameters(
        eqx_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    n_reps = _count_replicates(model)
    return model, task, n_reps


def _count_replicates(model) -> int:
    """Infer number of replicates from first batched leaf with ndim >= 3."""
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim >= 3:
            return int(leaf.shape[0])
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim == 2:
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



# ---------------------------------------------------------------------------
# Per-cell aggregate statistics
# ---------------------------------------------------------------------------

def compute_cell_stats(km: dict[str, np.ndarray]) -> dict:
    """Aggregate kinematics to per-replicate scalars then per-cell stats."""
    peak_vel_per_rep = km["peak_forward_velocity"].mean(axis=-1)   # (n_rep,)
    hold_drift_per_rep = km["hold_drift_mm"].mean(axis=-1)          # (n_rep,)
    ttp_per_rep = km["time_to_peak"].mean(axis=-1)                   # (n_rep,)

    mean_pv = float(peak_vel_per_rep.mean())
    sd_pv = float(peak_vel_per_rep.std(ddof=1)) if len(peak_vel_per_rep) > 1 else 0.0
    cv_peak_vel = sd_pv / mean_pv if mean_pv > 0 else float("nan")

    return {
        "peak_vel_per_rep": peak_vel_per_rep.tolist(),
        "hold_drift_per_rep": hold_drift_per_rep.tolist(),
        "time_to_peak_per_rep": ttp_per_rep.tolist(),
        "mean_peak_velocity": mean_pv,
        "sd_peak_velocity": sd_pv,
        "cv_peak_vel": cv_peak_vel,
        "mean_hold_drift_mm": float(hold_drift_per_rep.mean()),
        "sd_hold_drift_mm": float(hold_drift_per_rep.std(ddof=1)) if len(hold_drift_per_rep) > 1 else 0.0,
        "mean_time_to_peak_steps": float(ttp_per_rep.mean()),
    }


# ---------------------------------------------------------------------------
# RMSE-ratio computation (primary variance metric)
# ---------------------------------------------------------------------------

def _mean_pairwise_rmse(profiles_a: np.ndarray, profiles_b: np.ndarray) -> float:
    """Mean pairwise RMSE between every replicate in A and every in B.

    Uses nanmean so NaN-padded columns from `align_trials` are ignored
    (Bug: 06f7faf).
    """
    n_a = profiles_a.shape[0]
    n_b = profiles_b.shape[0]
    rmse_vals = []
    for i in range(n_a):
        for j in range(n_b):
            diff = profiles_a[i] - profiles_b[j]
            rmse_vals.append(float(np.sqrt(np.nanmean(diff ** 2))))
    return float(np.mean(rmse_vals)) if rmse_vals else float("nan")


def _within_cell_mean_pairwise_rmse(profiles: np.ndarray) -> float:
    """Mean pairwise RMSE between all distinct pairs within one cell.

    Uses nanmean so NaN-padded columns from `align_trials` are ignored
    (Bug: 06f7faf).
    """
    n_rep = profiles.shape[0]
    rmse_vals = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            diff = profiles[i] - profiles[j]
            rmse_vals.append(float(np.sqrt(np.nanmean(diff ** 2))))
    return float(np.mean(rmse_vals)) if rmse_vals else float("nan")


def compute_rmse_ratios(cell_kms: dict[str, dict]) -> dict[str, dict]:
    """Compute per-cell pairwise profile-RMSE ratio (within / nearest-across).

    For each cell i:
      within_rmse_vel[i]  = mean pairwise RMSE of velocity profiles across
                            all C(n_rep, 2) replicate pairs within cell i.
      across_rmse_vel[i]  = mean pairwise RMSE between replicates of cell i
                            and each other cell j; take the minimum over j != i
                            (nearest-neighbor across-cell RMSE).
      vel_rmse_ratio[i]   = within_rmse_vel[i] / across_rmse_vel[i]  (PRIMARY)

    Same computation for position profiles (pos_rmse_ratio, secondary).
    """
    labels = list(cell_kms.keys())

    # Bug: 06f7faf — align per-trial profiles to each trial's go cue BEFORE
    # the trial-axis collapse. Replicate-mean curves are computed via
    # `replicate_mean_curves` (nanmean over trial axis) so NaN padding from
    # trials with shorter pre/post windows doesn't bias the per-rep curve.
    # Use trim=False here because the cross-cell pairwise RMSE downstream
    # already handles NaN columns via nanmean and requires identical step
    # axes across cells (which a per-cell trim could disturb if go_idx
    # distributions differ).
    vel_profiles: dict[str, np.ndarray] = {}
    pos_profiles: dict[str, np.ndarray] = {}
    for label in labels:
        km = cell_kms[label]
        aligned_v, _c = align_trials(km["forward_vel_profile"], km["go_idx"])
        aligned_p, _c = align_trials(km["pos_forward_profile"], km["go_idx"])
        vel_profiles[label] = replicate_mean_curves(aligned_v, trim=False)
        pos_profiles[label] = replicate_mean_curves(aligned_p, trim=False)

    # NaN handling: replicate_mean_curves may leave NaN at extreme columns where
    # no trial contributed. _mean_pairwise_rmse uses (a-b)**2 and np.mean; mask
    # NaN columns to avoid propagating them through the RMSE calculation.
    def _finite_pair_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.isfinite(a) & np.isfinite(b)

    results: dict[str, dict] = {}
    for label in labels:
        vel_within = _within_cell_mean_pairwise_rmse(vel_profiles[label])
        pos_within = _within_cell_mean_pairwise_rmse(pos_profiles[label])

        vel_across_all = []
        pos_across_all = []
        for other in labels:
            if other == label:
                continue
            vel_across_all.append(
                _mean_pairwise_rmse(vel_profiles[label], vel_profiles[other])
            )
            pos_across_all.append(
                _mean_pairwise_rmse(pos_profiles[label], pos_profiles[other])
            )

        if vel_across_all:
            vel_nearest_across = float(min(vel_across_all))
            pos_nearest_across = float(min(pos_across_all))
        else:
            vel_nearest_across = float("nan")
            pos_nearest_across = float("nan")

        vel_rmse_ratio = vel_within / vel_nearest_across if vel_nearest_across > 0 else float("nan")
        pos_rmse_ratio = pos_within / pos_nearest_across if pos_nearest_across > 0 else float("nan")

        results[label] = {
            "vel_within_rmse": vel_within,
            "vel_nearest_across_rmse": vel_nearest_across,
            "vel_rmse_ratio": vel_rmse_ratio,         # PRIMARY metric
            "pos_within_rmse": pos_within,
            "pos_nearest_across_rmse": pos_nearest_across,
            "pos_rmse_ratio": pos_rmse_ratio,         # secondary
        }

    return results


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def make_peak_velocity_figure(cell_stats: dict[str, dict]) -> go.Figure:
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
            "Peak forward velocity per replicate — lit-replication 6-cell matrix (f47abb1)<br>"
            "<sup>CV (SD/mean of peak vel scalar) annotated — auxiliary metric. "
            "See rmse_ratio_comparison for primary metric.</sup>"
        ),
        yaxis_title="Peak forward velocity (m/s)",
        xaxis_title="Cell",
        width=1000,
        height=500,
        showlegend=False,
        margin=dict(l=70, r=40, t=80, b=60),
    )

    for i, label in enumerate(CELL_LABELS):
        if label not in cell_stats:
            continue
        stats = cell_stats[label]
        cv = stats["cv_peak_vel"]
        fig.add_annotation(
            x=i,
            y=max(stats["peak_vel_per_rep"]) * 1.05,
            text=f"CV={cv:.3f}",
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

    # Bug: 06f7faf — shared y-axes via profile_comparison_grid so per-replicate
    # traces are directly comparable across cells.
    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        vertical_spacing=0.06,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        v_fwd = km["forward_vel_profile"]  # (n_rep, n_trials, n_steps)
        go_idx = km["go_idx"]               # (n_trials,)
        n_rep, n_trials, n_steps = v_fwd.shape
        color = CELL_COLORS[CELL_LABELS.index(label) % len(CELL_COLORS)]

        # Bug: 06f7faf — go-cue alignment per trial; one curve per replicate
        # (replicate-level nanmean over aligned trials, trimmed to the
        # full-support column window).
        aligned_v, center = align_trials(v_fwd, go_idx)
        per_rep_curves, sl = replicate_mean_curves(aligned_v)  # (n_rep, n_kept_steps)
        t = ((np.arange(aligned_v.shape[-1]) - center) * dt)[sl]

        for rep in range(n_rep):
            fig.add_trace(go.Scatter(
                x=t,
                y=per_rep_curves[rep],
                mode="lines",
                name=f"Rep {rep}",
                line=dict(color=_color_rgba(color, 0.7), width=1.5),
                showlegend=(row == 1),
                legendgroup=f"rep{rep}",
            ), row=row, col=1)

        # Go cue lives at t=0 by construction
        fig.add_vline(
            x=0.0,
            line=dict(color="black", dash="dash", width=1),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=(
            "Forward velocity profiles (go-cue-aligned, replicate-mean curves) — "
            "lit-replication 6-cell (f47abb1)"
        ),
        width=900,
        height=220 * n_cells + 100,
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
    """Pre-go forward position (hold drift) per cell, one trace per replicate."""
    labels_present = [l for l in CELL_LABELS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    # Bug: 06f7faf — shared y-axes via profile_comparison_grid.
    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        vertical_spacing=0.06,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        pos_fwd = km["pos_forward_profile"]  # (n_rep, n_trials, n_steps)
        go_idx = km["go_idx"]                # (n_trials,)
        n_rep, n_trials, n_steps = pos_fwd.shape
        color = CELL_COLORS[CELL_LABELS.index(label) % len(CELL_COLORS)]

        # Bug: 06f7faf — go-cue alignment per trial; replicate-mean curves on
        # the full-support window. Then clip to [t <= 0] for the pre-go drift
        # figure.
        aligned_p, center = align_trials(pos_fwd, go_idx)
        per_rep_curves, sl = replicate_mean_curves(aligned_p)
        per_rep_curves = per_rep_curves * 1000.0  # mm
        t_rel = ((np.arange(aligned_p.shape[-1]) - center) * dt)[sl]
        keep = t_rel <= 0.0
        t_pre = t_rel[keep]
        pos_mm = per_rep_curves[:, keep]

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

        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1), row=row, col=1)

    fig.update_layout(
        title=(
            "Pre-go forward position drift (go-cue-aligned, per-replicate) — "
            "lit-replication 6-cell (f47abb1)"
        ),
        width=900,
        height=220 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd pos (mm)", row=row, col=1)

    return fig


def make_rmse_ratio_figure(
    rmse_ratios: dict[str, dict],
    cell_stats: dict[str, dict],
) -> go.Figure:
    """Grouped bar chart: velocity-RMSE ratio and position-RMSE ratio per cell.

    Primary metric (velocity-RMSE ratio) is the main bar; position-RMSE ratio
    is a secondary bar. The 0.5 target threshold is annotated as a dashed line.
    """
    labels_present = [l for l in CELL_LABELS if l in rmse_ratios]
    display_names = [CELL_DISPLAY_NAMES[l] for l in labels_present]

    vel_ratios = [rmse_ratios[l]["vel_rmse_ratio"] for l in labels_present]
    pos_ratios = [rmse_ratios[l]["pos_rmse_ratio"] for l in labels_present]
    cv_vals = [cell_stats[l]["cv_peak_vel"] for l in labels_present if l in cell_stats]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Vel-RMSE ratio (PRIMARY)",
        x=display_names,
        y=vel_ratios,
        marker_color="#1f77b4",
        text=[f"{v:.3f}" for v in vel_ratios],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Pos-RMSE ratio (secondary)",
        x=display_names,
        y=pos_ratios,
        marker_color="#ff7f0e",
        text=[f"{v:.3f}" for v in pos_ratios],
        textposition="outside",
    ))
    if cv_vals and len(cv_vals) == len(labels_present):
        fig.add_trace(go.Bar(
            name="CV peak vel (auxiliary)",
            x=display_names,
            y=cv_vals,
            marker_color="#2ca02c",
            opacity=0.6,
            text=[f"{v:.3f}" for v in cv_vals],
            textposition="outside",
        ))

    # Threshold line at 0.5
    fig.add_hline(
        y=0.5,
        line=dict(color="red", dash="dash", width=2),
        annotation_text="threshold 0.5",
        annotation_position="top right",
    )
    # Prior best line at 0.758
    fig.add_hline(
        y=0.758,
        line=dict(color="grey", dash="dot", width=1.5),
        annotation_text="prior best 0.758 (2bc95fd gru__jerk)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=(
            "Pairwise profile-RMSE ratio per cell — lit-replication 6-cell (f47abb1)<br>"
            "<sup>PRIMARY: velocity-RMSE ratio (within-cell / nearest-across-cell). "
            "Target: < 0.5 | Prior best (baseline GRU/jerk): 0.758</sup>"
        ),
        barmode="group",
        yaxis_title="RMSE ratio (within / nearest-across)",
        xaxis_title="Cell",
        width=1100,
        height=550,
        margin=dict(l=70, r=40, t=100, b=80),
        legend=dict(x=0.01, y=0.99),
    )

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

    artifact_base = args.artifact_base or (REPO_ROOT / "_artifacts" / EXPERIMENT / "runs")
    results_base = REPO_ROOT / "results" / EXPERIMENT

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")

    notes_dir = results_base / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

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
            f"CV={stats['cv_peak_vel']:.3f}  "
            f"hold_drift={stats['mean_hold_drift_mm']:.2f} +/- {stats['sd_hold_drift_mm']:.2f} mm"
        )

    if not cell_stats:
        print("\nNo cells loaded -- aborting.")
        return

    # -----------------------------------------------------------------------
    # RMSE ratios (primary variance metric -- cross-cell computation)
    # -----------------------------------------------------------------------
    print("\n--- Computing pairwise RMSE ratios (primary metric) ---")
    rmse_ratios: dict[str, dict] = {}
    if len(cell_kms) >= 2:
        rmse_ratios = compute_rmse_ratios(cell_kms)
        for label in CELL_LABELS:
            if label not in rmse_ratios:
                continue
            r = rmse_ratios[label]
            print(
                f"  [{label}] vel-RMSE-ratio={r['vel_rmse_ratio']:.3f}  "
                f"(within={r['vel_within_rmse']:.4f}, "
                f"nearest-across={r['vel_nearest_across_rmse']:.4f})  "
                f"pos-RMSE-ratio={r['pos_rmse_ratio']:.3f}"
            )
    else:
        print("  Less than 2 cells loaded -- cannot compute cross-cell RMSE ratios.")

    # -----------------------------------------------------------------------
    # Figures (HTML only -- no PNG renders)
    # -----------------------------------------------------------------------
    print("\n--- Building figures ---")

    # Figure 1: Peak velocity distributions
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
        },
        "metric_note": (
            "Annotation shows CV (SD/mean of peak vel scalar) -- auxiliary metric. "
            "Primary metric is vel_rmse_ratio in rmse_ratio_comparison figure."
        ),
        "cell_stats": {
            label: {
                "mean_peak_velocity": stats["mean_peak_velocity"],
                "sd_peak_velocity": stats["sd_peak_velocity"],
                "cv_peak_vel": stats["cv_peak_vel"],
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
    print(f"  Spec: {pv_out['spec_path']}")
    print(f"  Render: {pv_out['render_path']}")

    # Figure 2: Forward velocity profiles
    if cell_kms:
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
                {"name": "replicate_nanmean_over_trials", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_LABELS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "dt": 0.01,
                "alignment": "go_cue_per_trial",
                "shared_yaxes": "all",
            },
            "fix_note": "Bug: 06f7faf — go-cue alignment + trim-to-full-support + shared y-axes across cells.",
        }
        fv_out = save_figure(
            fig=fig_fv, spec=spec_fv,
            package="rlrmp", experiment=EXPERIMENT, topic="forward_velocity_profiles",
            extra_packages=["rlrmp"],
        )
        print(f"  Spec: {fv_out['spec_path']}")
        print(f"  Render: {fv_out['render_path']}")

    # Figure 3: Hold drift profiles
    if cell_kms:
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
                {"name": "replicate_nanmean_over_trials", "kwargs": {}},
                {"name": "clip_to_pre_go_window", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_LABELS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "dt": 0.01,
                "alignment": "go_cue_per_trial",
                "shared_yaxes": "all",
            },
            "fix_note": "Bug: 06f7faf — go-cue alignment fix + shared y-axes across cells (per-replicate variant retains full aligned window).",
        }
        hd_out = save_figure(
            fig=fig_hd, spec=spec_hd,
            package="rlrmp", experiment=EXPERIMENT, topic="hold_drift_profiles",
            extra_packages=["rlrmp"],
        )
        print(f"  Spec: {hd_out['spec_path']}")
        print(f"  Render: {hd_out['render_path']}")

    # Figure 4: RMSE ratio comparison (PRIMARY metric)
    if rmse_ratios and cell_stats:
        fig_rmse = make_rmse_ratio_figure(rmse_ratios, cell_stats)
        spec_rmse = {
            "figure_kind": "rmse_ratio_comparison_bar",
            "experiment": EXPERIMENT,
            "metric_description": (
                "Primary: velocity-RMSE ratio = within-cell mean pairwise RMSE / "
                "nearest-across-cell mean pairwise RMSE on forward-velocity profiles. "
                "Secondary: same on forward-position profiles. "
                "Auxiliary: CV = SD(peak_vel) / mean(peak_vel) across replicates. "
                "Target threshold 0.5; prior best (baseline GRU/jerk, 2bc95fd): 0.758."
            ),
            "inputs": input_artifacts,
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
                {"name": "replicate_nanmean_over_trials", "kwargs": {}},
                {"name": "pairwise_profile_rmse_ratio_velocity", "kwargs": {}},
                {"name": "pairwise_profile_rmse_ratio_position", "kwargs": {}},
            ],
            "fix_note": "Bug: 06f7faf — go-cue alignment fix; RMSE now computed on go-aligned per-rep curves.",
            "plot_kwargs": {
                "cells": CELL_LABELS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
            },
            "rmse_ratios": {
                label: {
                    "vel_within_rmse": r["vel_within_rmse"],
                    "vel_nearest_across_rmse": r["vel_nearest_across_rmse"],
                    "vel_rmse_ratio": r["vel_rmse_ratio"],
                    "pos_within_rmse": r["pos_within_rmse"],
                    "pos_nearest_across_rmse": r["pos_nearest_across_rmse"],
                    "pos_rmse_ratio": r["pos_rmse_ratio"],
                }
                for label, r in rmse_ratios.items()
            },
        }
        rmse_out = save_figure(
            fig=fig_rmse, spec=spec_rmse,
            package="rlrmp", experiment=EXPERIMENT, topic="rmse_ratio_comparison",
            extra_packages=["rlrmp"],
        )
        print(f"  Spec: {rmse_out['spec_path']}")
        print(f"  Render: {rmse_out['render_path']}")

    # -----------------------------------------------------------------------
    # Summary table + decision
    # -----------------------------------------------------------------------
    print("\n=== VARIANCE ANALYSIS SUMMARY (f47abb1 lit-replication) ===\n")
    prior_best_vr = 0.758  # GRU/jerk baseline_jerk_vrnn_matrix
    winner_threshold = 0.5
    winners = []

    header = (
        f"{'Cell':28s} {'Vel-RMSE-ratio':>15} {'Pos-RMSE-ratio':>15} "
        f"{'CV (peak vel)':>14} {'Mean PV (m/s)':>14} {'Hold drift (mm)':>16} {'TTP':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for label in CELL_LABELS:
        if label not in cell_stats:
            print(f"  {CELL_DISPLAY_NAMES[label]:28s} SKIPPED")
            continue
        stats = cell_stats[label]
        r = rmse_ratios.get(label, {})
        vel_rr = r.get("vel_rmse_ratio", float("nan"))
        pos_rr = r.get("pos_rmse_ratio", float("nan"))
        cv = stats["cv_peak_vel"]
        if not np.isnan(vel_rr) and vel_rr < winner_threshold:
            winners.append(label)
        flag = " <-- WINNER" if (not np.isnan(vel_rr) and vel_rr < winner_threshold) else ""
        print(
            f"  {CELL_DISPLAY_NAMES[label]:28s} "
            f"{vel_rr:>15.3f} "
            f"{pos_rr:>15.3f} "
            f"{cv:>14.3f} "
            f"{stats['mean_peak_velocity']:>14.4f} "
            f"{stats['mean_hold_drift_mm']:>16.3f} "
            f"{stats['mean_time_to_peak_steps']:>6.1f}"
            f"{flag}"
        )

    print(sep)
    print(
        f"\nPrimary decision criterion: vel-RMSE-ratio < {winner_threshold} "
        f"(prior best = {prior_best_vr} from 2bc95fd GRU/jerk)"
    )
    if winners:
        print(f"WINNERS ({len(winners)} cells beat threshold):")
        for w in winners:
            r = rmse_ratios.get(w, {})
            print(f"  {CELL_DISPLAY_NAMES[w]}: vel-RMSE-ratio = {r.get('vel_rmse_ratio', float('nan')):.3f}")
    else:
        candidate_labels = [l for l in cell_stats if l in rmse_ratios]
        best_label = min(
            candidate_labels,
            key=lambda l: rmse_ratios[l].get("vel_rmse_ratio", float("inf")),
            default=None,
        )
        if best_label:
            r = rmse_ratios.get(best_label, {})
            print(
                f"No cell met threshold (vel-RMSE ratio < {winner_threshold}). "
                f"Best: {CELL_DISPLAY_NAMES[best_label]} "
                f"vel-RMSE-ratio={r.get('vel_rmse_ratio', float('nan')):.3f} "
                f"(target was <{winner_threshold}, prior best was {prior_best_vr})"
            )
        else:
            print("No cells evaluated.")

    # -----------------------------------------------------------------------
    # Write analysis notes
    # -----------------------------------------------------------------------
    notes_path = notes_dir / "variance_analysis.md"
    lines = [
        "# Variance Analysis — Lit-Replication 6-Cell Matrix (f47abb1)",
        "",
        "## Setup",
        "",
        "This matrix tests whether faithful replication of the Chaisanguanthum & Shenoy",
        "2019 (C&S) loss schedule produces models with systematically better velocity-RMSE",
        "ratios and lower inter-replicate variance than the current production loss.",
        "",
        "Two design dimensions crossed:",
        "1. **Jerk regulariser** (`nn_output_jerk`): on (1e5) vs off (0.0).",
        "   Shahbazi et al. 2025 Eq. 1 used jerk; C&S 2019 did not.",
        "2. **Position schedule**: flat / post-go `(t/T)^6` / full-trial `(t/T)^6`.",
        "   The powerlaw concentrates ~98% of position weight in the last 30% of the trial.",
        "",
        "**Corrected hold-penalty bug** (vs 2bc95fd): prior run used `==` check for",
        "`center_out_delayed_reach` task type, which silently failed. Fixed in commit `22153e4`.",
        "This run applies hold penalties correctly for the first time.",
        "",
        "**Corrected `nn_hidden_derivative` weight**: this matrix does not set",
        "`nn_hidden_derivative`; it uses the default of 0.0. The prior 2bc95fd matrix",
        "used `nn_hidden_derivative=1e2` in the `gru__jerk_smooth_high` and combo cells,",
        "but that is not part of the lit-replication design.",
        "",
        "### Run metadata",
        "",
        f"- Experiment hash: `{EXPERIMENT}`",
        "- SISU: 0.5",
        "- Perturbation: 0 (clean reach)",
        "- Validation trials: 8 center-out reach directions",
        "- Pod: jmhwbqd61kw9z3, RTX 4090, CZ datacenter",
        "- Wall-clock: ~32 min/cell, 6 cells sequential",
        "- Git SHA: 15f647bfbcb8df20966e94141667ee41f24af5fe",
        "",
        "## Metrics",
        "",
        "**Primary: velocity-RMSE ratio** — within-cell mean pairwise RMSE on the",
        "forward-velocity profile / nearest-neighbor across-cell mean pairwise RMSE.",
        "Matches the prior `baseline_jerk_vrnn_matrix` metric.",
        "Prior best (GRU/jerk, 2bc95fd): 0.758.",
        "Decision threshold: < 0.50.",
        "",
        "**Secondary: position-RMSE ratio** — same computation on forward-position profile.",
        "",
        "**Auxiliary: CV (SD/mean of peak vel)** — scalar summary of replicate spread on",
        "peak velocity. Reported for completeness; does NOT drive the decision.",
        "",
        "**IMPORTANT — cross-schedule comparisons**: absolute loss values are NOT",
        "comparable across position schedule shapes. The powerlaw `(t/T)^6` concentrates",
        "~98% of position weight in the last 30% of the trial, making the weighted sum",
        "structurally lower than for flat. Compare WITHIN schedule shape only:",
        "  - Flat: `lit__flat_jerk` vs `lit__flat_nojerk`",
        "  - Post-go: `lit__post_jerk` vs `lit__post_nojerk`",
        "  - Full-trial: `lit__full_jerk` vs `lit__full_nojerk`",
        "",
        "## Results Table",
        "",
        "| Cell | Display Name | Vel-RMSE ratio (PRIMARY) | Pos-RMSE ratio | "
        "CV (peak vel) | Mean PV (m/s) | SD PV (m/s) | Hold Drift (mm) | TTP (steps) |",
        "|------|------|---------|---------|---------|---------|---------|---------|---------|",
    ]

    for label in CELL_LABELS:
        if label not in cell_stats:
            lines.append(f"| {label} | {CELL_DISPLAY_NAMES[label]} | SKIPPED | - | - | - | - | - | - |")
            continue
        stats = cell_stats[label]
        r = rmse_ratios.get(label, {})
        vel_rr = r.get("vel_rmse_ratio", float("nan"))
        pos_rr = r.get("pos_rmse_ratio", float("nan"))
        cv = stats["cv_peak_vel"]
        winner_flag = " *" if (not np.isnan(vel_rr) and vel_rr < winner_threshold) else ""
        vel_rr_str = f"{vel_rr:.3f}{winner_flag}" if not np.isnan(vel_rr) else "n/a"
        pos_rr_str = f"{pos_rr:.3f}" if not np.isnan(pos_rr) else "n/a"
        lines.append(
            f"| {label} | {CELL_DISPLAY_NAMES[label]} "
            f"| {vel_rr_str} "
            f"| {pos_rr_str} "
            f"| {cv:.3f} "
            f"| {stats['mean_peak_velocity']:.4f} "
            f"| {stats['sd_peak_velocity']:.4f} "
            f"| {stats['mean_hold_drift_mm']:.3f} +/- {stats['sd_hold_drift_mm']:.3f} "
            f"| {stats['mean_time_to_peak_steps']:.1f} |"
        )

    lines += [
        "",
        "\\* = beats primary threshold (vel-RMSE ratio < 0.50).",
        "",
        "## RMSE Detail (within vs across)",
        "",
        "| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) | "
        "Pos within-RMSE (m) | Pos nearest-across-RMSE (m) |",
        "|------|---------|---------|---------|---------|",
    ]
    for label in CELL_LABELS:
        if label not in rmse_ratios:
            continue
        r = rmse_ratios[label]
        lines.append(
            f"| {label} "
            f"| {r['vel_within_rmse']:.4f} "
            f"| {r['vel_nearest_across_rmse']:.4f} "
            f"| {r['pos_within_rmse']:.4f} "
            f"| {r['pos_nearest_across_rmse']:.4f} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        f"**Primary threshold**: vel-RMSE ratio < {winner_threshold}",
        f"Prior best (GRU/jerk, 2bc95fd): {prior_best_vr}",
        "",
    ]

    if winners:
        lines.append(f"**{len(winners)} cell(s) met the threshold:**")
        lines.append("")
        for w in winners:
            stats = cell_stats[w]
            r = rmse_ratios.get(w, {})
            lines.append(
                f"- **{CELL_DISPLAY_NAMES[w]}** (`{w}`): "
                f"vel-RMSE-ratio = {r.get('vel_rmse_ratio', float('nan')):.3f}, "
                f"CV = {stats['cv_peak_vel']:.3f}, "
                f"mean PV = {stats['mean_peak_velocity']:.4f} m/s, "
                f"hold drift = {stats['mean_hold_drift_mm']:.3f} mm"
            )
    else:
        candidate_labels = [l for l in cell_stats if l in rmse_ratios]
        best_label = min(
            candidate_labels,
            key=lambda l: rmse_ratios[l].get("vel_rmse_ratio", float("inf")),
            default=None,
        )
        if best_label:
            r = rmse_ratios.get(best_label, {})
            lines.append("**No cell met the threshold.**")
            lines.append("")
            lines.append(
                f"Best cell: **{CELL_DISPLAY_NAMES[best_label]}** (`{best_label}`) "
                f"with vel-RMSE-ratio = {r.get('vel_rmse_ratio', float('nan')):.3f}"
            )
        else:
            lines.append("No cells evaluated.")

    lines += [
        "",
        "## Per-axis findings",
        "",
        "### Jerk axis (compare within same schedule shape)",
        "",
        "Within each schedule shape, compare jerk-on vs jerk-off:",
        "  - Flat: `lit__flat_jerk` vs `lit__flat_nojerk`",
        "  - Post-go PL: `lit__post_jerk` vs `lit__post_nojerk`",
        "  - Full-trial PL: `lit__full_jerk` vs `lit__full_nojerk`",
        "",
        "(Results table populated after script run; see above.)",
        "",
        "### Position schedule axis (compare within same jerk condition)",
        "",
        "Within each jerk condition, compare flat vs post-go PL vs full-trial PL:",
        "  - Jerk on: `lit__flat_jerk` vs `lit__post_jerk` vs `lit__full_jerk`",
        "  - Jerk off: `lit__flat_nojerk` vs `lit__post_nojerk` vs `lit__full_nojerk`",
        "",
        "(Results table populated after script run; see above.)",
        "",
        "## Conditional follow-up triggers (per f47abb1 issue body)",
        "",
        "**Pre-go-mask follow-up**: if jerk-disabled cells (lit__flat_nojerk,",
        "lit__post_nojerk, lit__full_nojerk) show significant anticipation relative to",
        "jerk-enabled cells (hold drift > 1 mm or visible pre-go velocity ramp), reintroduce",
        "`--nn-output-pre-go` as a follow-up matrix lever (suggested starting weight: 1e-2).",
        "",
        "## Anticipation (Hold Drift)",
        "",
        "Hold drift = max forward displacement (toward target, in mm) before the go cue.",
        "Positive = anticipatory movement. Threshold for 'good hold': < 0.5 mm.",
        "",
    ]

    for label in CELL_LABELS:
        if label not in cell_stats:
            continue
        stats = cell_stats[label]
        flag = ""
        drift = stats["mean_hold_drift_mm"]
        if drift > 1.0:
            flag = " <-- anticipation trigger (> 1 mm)"
        elif drift < 0.5:
            flag = " (good hold)"
        lines.append(
            f"- {CELL_DISPLAY_NAMES[label]} (`{label}`): "
            f"{drift:.3f} +/- {stats['sd_hold_drift_mm']:.3f} mm{flag}"
        )

    lines += [
        "",
        "## Figures",
        "",
        f"- `results/{EXPERIMENT}/figures/rmse_ratio_comparison/` — Bar chart (PRIMARY)",
        f"- `results/{EXPERIMENT}/figures/peak_velocity_distributions/` — Violin (CV annotated, auxiliary)",
        f"- `results/{EXPERIMENT}/figures/forward_velocity_profiles/` — Velocity time series per cell",
        f"- `results/{EXPERIMENT}/figures/hold_drift_profiles/` — Pre-go forward position (anticipation)",
        "",
        "HTML renders in `_artifacts/f47abb1/figures/<name>/figure.html`.",
    ]

    # Bug: 06f7faf — use update_marked_section so hand-edited preambles (e.g.
    # "Corrected after go-cue alignment fix") are preserved on re-run.
    update_marked_section(notes_path, "variance_analysis", "\n".join(lines) + "\n")
    print(f"\nSaved analysis notes: {notes_path}")

    # Save per-cell stats as JSON for downstream use
    stats_json_path = notes_dir / "variance_analysis_data.json"
    json_data = {
        "experiment": EXPERIMENT,
        "sisu": args.sisu,
        "primary_metric": "vel_rmse_ratio",
        "prior_best_vel_rmse_ratio": prior_best_vr,
        "winner_threshold": winner_threshold,
        "winners": winners,
        "cells": cell_stats,
        "rmse_ratios": {
            label: {k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                    for k, v in r.items()}
            for label, r in rmse_ratios.items()
        },
    }
    with open(stats_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved stats JSON: {stats_json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
