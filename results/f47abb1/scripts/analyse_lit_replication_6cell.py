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
from rlrmp.analysis.multi_cell_driver import (
    args_namespace,
    run_replicate_kinematics_analysis,
)
from rlrmp.viz.colors import hex_to_rgba as _color_rgba

import argparse
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.math.trial_alignment import (
    align_trials,
    replicate_mean_curves,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.paths import REPO_ROOT  # Bug: 8404108 — was __file__-relative
from rlrmp.train.minimax import build_hps
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz.figures import (
    build_forward_velocity_figure as canonical_forward_velocity_figure,
    build_hold_drift_figure as canonical_hold_drift_figure,
)

warnings.filterwarnings("ignore")

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
    return args_namespace(
        profile="lit_replication",
        n_warmup_batches=N_WARMUP_BATCHES,
        n_replicates=N_REPLICATES,
        overrides=CELL_EXTRA_ARGS[label],
    )


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
    colors = {
        label: CELL_COLORS[index % len(CELL_COLORS)]
        for index, label in enumerate(CELL_LABELS)
    }
    return canonical_forward_velocity_figure(
        cell_kms,
        labels=CELL_LABELS,
        display_names=CELL_DISPLAY_NAMES,
        colors=colors,
        trace_mode="replicate",
        title=(
            "Forward velocity profiles (go-cue-aligned, replicate-mean curves) — "
            "lit-replication 6-cell (f47abb1)"
        ),
        width=900,
        height_per_cell=220,
        vertical_spacing=0.06,
        dt=dt,
    )

def make_hold_drift_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Pre-go forward position (hold drift) per cell, one trace per replicate."""
    colors = {
        label: CELL_COLORS[index % len(CELL_COLORS)]
        for index, label in enumerate(CELL_LABELS)
    }
    return canonical_hold_drift_figure(
        cell_kms,
        labels=CELL_LABELS,
        display_names=CELL_DISPLAY_NAMES,
        colors=colors,
        trace_mode="replicate",
        title=(
            "Pre-go forward position drift (go-cue-aligned, per-replicate) — "
            "lit-replication 6-cell (f47abb1)"
        ),
        width=900,
        height_per_cell=220,
        vertical_spacing=0.06,
        dt=dt,
    )

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

def main() -> None:
    """Run the configured multi-cell kinematics analysis."""

    run_replicate_kinematics_analysis(
        {
            "__doc__": __doc__,
            "REPO_ROOT": REPO_ROOT,
            "EXPERIMENT": EXPERIMENT,
            "CELL_LABELS": CELL_LABELS,
            "CELL_DISPLAY_NAMES": CELL_DISPLAY_NAMES,
            "N_REPLICATES": N_REPLICATES,
            "load_cell_model": load_cell_model,
            "build_zero_pert_trials": build_zero_pert_trials,
            "eval_ensemble": eval_ensemble,
            "compute_cell_stats": compute_cell_stats,
            "compute_rmse_ratios": compute_rmse_ratios,
            "make_peak_velocity_figure": make_peak_velocity_figure,
            "make_forward_velocity_profile_figure": make_forward_velocity_profile_figure,
            "make_hold_drift_figure": make_hold_drift_figure,
            "make_rmse_ratio_figure": make_rmse_ratio_figure,
        },
        profile="lit_replication",
    )


if __name__ == "__main__":
    main()
